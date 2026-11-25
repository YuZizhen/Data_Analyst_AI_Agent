"""Deterministic evidence checks used by the analysis graph before publishing a reply."""

import re
from collections import defaultdict
from hashlib import sha256
from math import fsum, isclose, isfinite
from statistics import StatisticsError, correlation, mean, median

import pandas as pd
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException

from src.tools.dataset import finite_numbers
from src.tools.preprocessing import PREPARATION_FUNCTIONS, filter_frame

NUMERIC_TOOLS = {
    "average_by_group", "group_report", "trend_report", "distribution_report",
    "rank_relationships", "relationship_report", "group_distribution_report", "feature_importance_report", "evaluate_prediction", "predict_value",
}
CLAIM = re.compile(r"导致|显著(?:影响|相关|提高|降低)|决定了|证明.{0,10}因果|\bcaus(?:e[ds]?|ally)\b|statistically significant", re.I)
RANK_OVERCLAIM = re.compile(r"(?:最为?显著|影响(?:最强|最大)|most significant|greatest impact)", re.I)
IMPORTANCE_OVERCLAIM = re.compile(
    r"(?:贡献|占).{0,25}(?:预测误差|总误差)(?!\s*(?:的)?(?:增量|增加|变化))"
    r"|account(?:s|ed)? for.{0,28}(?:prediction|total) error\b(?!\s*(?:increase|change))", re.I,
)
CAVEAT = re.compile(r"不(?:能|可|代表|说明|足以|证明|表示)|无(?:法|证据)|没有(?:证据|做|进行|证明)|未(?:经|做|检验|证明)|\b(?:not|cannot|no evidence|doesn.t)\b", re.I)


def frame_signature(frame: pd.DataFrame) -> str:
    digest = sha256(repr(list(zip(frame.columns, map(str, frame.dtypes)))).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def analysis_scope(frame: pd.DataFrame, prepared) -> dict:
    return {"rows": len(frame), "signature": frame_signature(frame), "steps": list(prepared.steps) if prepared else []}


def check(label: str, status: str, detail: str, *, tool: str = "") -> dict:
    return {"label": label, "status": status, "detail": detail, "tool": tool}


def replay_scope(source: pd.DataFrame, scope: dict) -> pd.DataFrame:
    frame = source
    for step in scope["steps"]:
        frame = PREPARATION_FUNCTIONS[step["tool"]](frame, **step["parameters"])
    return frame


def verify_requested_scope(source, frame, steps, inherited, request, tool, supplied_values):
    checks = []
    conditions = [item for item in request["conditions"] if item["source"] != "record_input"]
    unresolved = request["unresolved"]
    if unresolved:
        phrases = "、".join(f"「{item}」" for item in unresolved)
        checks.append(check("问题的筛选条件", "limited", f"以下条件未能完整映射为可核对的筛选规则：{phrases}。", tool=tool))
    initial = [] if request["reset"] else inherited
    prefix = len(initial) if steps[:len(initial)] == initial else 0
    actual = [step["parameters"] for step in steps if step["tool"] == "filter_rows"]
    comparison = {"requested": conditions, "actual": actual, "unresolved": unresolved, "actual_rows": len(frame)}
    if prefix:
        comparison["inherited"] = [step["parameters"] for step in steps[:prefix] if step["tool"] == "filter_rows"]
    new_filters = any(step["tool"] == "filter_rows" for step in steps[prefix:])
    if new_filters and not conditions and not unresolved and not request["reset"]:
        checks.append(check("请求与执行范围", "limited", "工具执行了新的筛选，但从用户原话未能独立确认这些筛选条件。", tool=tool))
    if supplied_values:
        comparison["prediction_inputs"] = supplied_values

    for item in conditions:
        field, operator, value = item["column_name"], item["operator"], item["value"]
        try:
            matching = len(filter_frame(frame, field, operator, value))
        except ToolException:
            checks.append(check("问题的筛选条件", "limited", f"无法按当前字段类型核对「{item['phrase']}」。", tool=tool))
            continue
        valid = matching == len(frame)
        checks.append(check("问题的筛选条件", "passed" if valid else "failed",
                            f"工具收到的 {len(frame):,} 行中，有 {matching:,} 行满足「{item['phrase']}」"
                            f"（{field} {operator} {value}）。", tool=tool))

    if not unresolved and not any(item["status"] == "limited" for item in checks):
        try:
            expected = replay_scope(source, {"steps": initial})
            current = [step for step in steps[prefix:] if step["tool"] != "filter_rows"]
            conversions = [step for step in current if step["tool"] == "convert_column"]
            expected = replay_scope(expected, {"steps": conversions})
            for item in conditions:
                expected = filter_frame(expected, item["column_name"], item["operator"], item["value"])
            expected = replay_scope(expected, {"steps": [step for step in current if step["tool"] != "convert_column"]})
            missing = len(expected.index.difference(frame.index))
            extra = len(frame.index.difference(expected.index))
            comparison.update(expected_rows=len(expected), missing_rows=missing, extra_rows=extra)
            duplicated = any(step["tool"] == "drop_duplicates" for step in current)
            if duplicated and (missing or extra):
                status = "limited"
                detail = "去重和筛选的顺序影响了保留的行，暂无法完整确认范围。"
            else:
                status = "failed" if missing or extra else "passed"
                detail = f"按请求应使用 {len(expected):,} 行，实际使用 {len(frame):,} 行。漏掉 {missing:,} 行，多纳入 {extra:,} 行。"
            checks.append(check("请求与执行范围", status, detail, tool=tool))
        except (KeyError, ToolException, TypeError, ValueError):
            checks.append(check("请求与执行范围", "limited", "无法结合已记录的清洗步骤重建完整的请求范围。", tool=tool))
    if not checks:
        checks.append(check("请求与执行范围", "limited", "没有足够信息核对筛选范围。", tool=tool))
    statuses = {item["status"] for item in checks}
    comparison["status"] = "failed" if "failed" in statuses else "limited" if "limited" in statuses else "passed"
    checks[-1]["scope_comparison"] = comparison
    return checks


def check_groups(frame: pd.DataFrame, args: dict, evidence: dict) -> tuple[bool, str]:
    aggregation = args.get("aggregation", "mean")
    values = [1] * len(frame) if aggregation == "count" else finite_numbers(frame[args["value_column"]])
    groups = defaultdict(list)
    for label, value in zip(frame[args["group_column"]], values):
        if pd.notna(value):
            groups[None if pd.isna(label) else label].append(float(value))
    calculate = {"mean": mean, "median": median, "sum": fsum, "count": len}[aggregation]
    expected = sorted(
        (("(Missing)" if name is None else str(name), calculate(items)) for name, items in groups.items()),
        key=lambda item: item[1], reverse=True,
    )
    ranking = expected[:args.get("top_n", 10)]
    reported = evidence["groups"]
    matches = (
        evidence["working_rows"] == len(frame) and evidence["group_count"] == len(expected)
        and evidence["rows_used"] == sum(map(len, groups.values())) and len(reported) == len(ranking)
        and all(isclose(row["value"], value, rel_tol=1e-9, abs_tol=1e-9) for row, (_, value) in zip(reported, ranking))
    )
    if matches:
        # Match a reported row at most once, including when the display label is ambiguous.
        remaining = expected.copy()
        for row in reported:
            index = next((i for i, (name, value) in enumerate(remaining) if row["group"] == name
                          and isclose(row["value"], value, rel_tol=1e-9, abs_tol=1e-9)), None)
            if index is None:
                matches = False
                break
            remaining.pop(index)
    example = "、".join(f"{name} {value:,.6g}" for name, value in ranking[:3])
    summary = f"用 Python 逐组重算了 {len(expected):,} 组，涉及 {sum(map(len, groups.values())):,} 行。"
    return matches, summary + (f"复算值示例：{example}。" if example else "")


def check_pair(frame: pd.DataFrame, args: dict, evidence: dict) -> tuple[bool, str]:
    pairs = pd.concat([finite_numbers(frame[args[key]]) for key in ("x_column", "y_column")], axis=1).dropna()
    try:
        expected = correlation(pairs.iloc[:, 0], pairs.iloc[:, 1])
        if not isfinite(expected):
            expected = None
    except StatisticsError:
        expected = None
    actual = evidence["r"]
    matches = (actual is None if expected is None else actual is not None and isclose(actual, expected, abs_tol=1e-9))
    value = "无法定义" if expected is None else f"{expected:+.3f}"
    return matches and len(pairs) == evidence["paired_rows"] and len(frame) == evidence["working_rows"], (
        f"用 Python 重算 {len(pairs):,} 对值的 Pearson r，结果为 {value}。"
    )


def check_distribution(frame: pd.DataFrame, args: dict, evidence: dict) -> tuple[bool, str]:
    values = finite_numbers(frame[args["value_column"]]).dropna().tolist()
    expected = {"median": median(values), "min": min(values), "max": max(values)}
    matches = len(frame) == evidence["working_rows"] and len(values) == evidence["rows_used"] and all(
        isclose(value, evidence[name], rel_tol=1e-9, abs_tol=1e-9) for name, value in expected.items()
    )
    return matches, f"独立核对了 {len(values):,} 个值，中位数 {expected['median']:,.6g}，范围 {expected['min']:,.6g} 至 {expected['max']:,.6g}。"


RECOMPUTE = {
    "group_summary": check_groups, "pairwise_correlation": check_pair, "distribution_summary": check_distribution,
}


def validate_results(source: pd.DataFrame, messages, normalize_parameters, rejected: set[str], recovery_start: int, emit,
                     *, request_scope: dict, input_preparation: list[dict]) -> tuple[list[dict], set[str]]:
    user = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), -1)
    current = messages[user + 1:]
    calls = {call["id"]: call for item in current if isinstance(item, AIMessage) for call in item.tool_calls}
    tools = [
        msg for msg in current if isinstance(msg, ToolMessage) and msg.name in NUMERIC_TOOLS
        and msg.status == "success" and msg.tool_call_id in calls and msg.tool_call_id not in rejected
    ]
    active_ids = {msg.tool_call_id for msg in tools}
    needs_replacement = {
        msg.name for msg in current if isinstance(msg, ToolMessage) and msg.tool_call_id in rejected
    } - {
        msg.name for msg in messages[max(recovery_start, user + 1):]
        if isinstance(msg, ToolMessage) and msg.tool_call_id in active_ids
    }
    if not tools and not needs_replacement:
        return [], set()

    checks, invalid = [], set()
    for name in sorted(needs_replacement):
        checks.append(check("重试结果", "failed", "上次的结果已被排除，本次还没有新的成功计算可核对。", tool=name))
    emit("scope")
    evidence_to_check = []
    for tool in tools:
        evidence = tool.artifact or {}
        supplied_values = evidence.get("prediction_evaluation", {}).get("estimate", {}).get("inputs", {})
        scope = evidence.get("analysis_scope")
        if scope is None:
            checks.append(check("数据范围", "limited", "这次调用缺少可核对的数据范围记录。", tool=tool.name))
            continue
        try:
            frame = replay_scope(source, scope)
            same_scope = len(frame) == scope["rows"] and frame_signature(frame) == scope["signature"]
        except (KeyError, ToolException, TypeError, ValueError):
            frame, same_scope = source, False
        if not same_scope:
            checks.append(check("数据范围", "failed", "按已记录的步骤重建的数据，与工具实际使用的数据不一致。", tool=tool.name))
            invalid.add(tool.tool_call_id)
            continue
        prediction = evidence.get("prediction_evaluation") or evidence.get("feature_importance")
        scope_detail = f"重建了 {len(frame):,} 行，核对行内容和字段类型，与工具收到的工作表一致。"
        if prediction:
            scope_detail += f"报告中的预测评估使用其中 {prediction['rows']:,} 行。"
        checks.append(check("数据范围", "passed", scope_detail, tool=tool.name))
        scope_checks = verify_requested_scope(source, frame, scope["steps"], input_preparation, request_scope, tool.name, supplied_values)
        checks.extend(scope_checks)
        if any(item["status"] == "failed" for item in scope_checks):
            invalid.add(tool.tool_call_id)
        evidence_to_check.append((tool, evidence, frame))
    emit("numbers")
    for tool, evidence, frame in evidence_to_check:
        for key, recompute in RECOMPUTE.items():
            if key in evidence:
                try:
                    call = calls[tool.tool_call_id]
                    matches, detail = recompute(frame, normalize_parameters(call["name"], call["args"]), evidence[key])
                except (ArithmeticError, KeyError, TypeError, ValueError):
                    matches, detail = False, "无法根据记录的参数和输入重算这项结果。"
                checks.append(check("独立复算", "passed" if matches else "failed", detail, tool=tool.name))
                if not matches:
                    invalid.add(tool.tool_call_id)
                break
        else:
            checks.append(check("独立复算", "limited", "这类计算目前已核对数据范围，独立复算尚未覆盖。", tool=tool.name))

    emit("conclusion")
    draft = current[-1].text if current and isinstance(current[-1], AIMessage) else ""
    clauses = re.split(r"[。！？\n；;]|但(?:是)?|不过|\b(?:but|however)\b", draft, flags=re.I)
    patterns = [CLAIM]
    if any(tool.name == "rank_relationships" for tool in tools):
        patterns.append(RANK_OVERCLAIM)
    if any(tool.name == "feature_importance_report" for tool in tools):
        patterns.append(IMPORTANCE_OVERCLAIM)
    flagged = any(pattern.search(part) and not CAVEAT.search(part) for pattern in patterns for part in clauses)
    if flagged:
        checks.append(check("结论措辞", "failed", "检查发现因果、显著性或误差占比表述可能超出证据。请描述样本关联或打乱字段后产生的正向误差增量占比。"))
    else:
        checks.append(check("结论措辞", "passed", "已扫描常见的因果和显著性断言。这项检查关注措辞边界。"))
    return checks, invalid
