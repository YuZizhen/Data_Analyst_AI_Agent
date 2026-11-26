"""Local browser interface for the data analyst agent."""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import streamlit as st
from langchain.messages import AIMessage, AnyMessage, HumanMessage

from src.agent import (
    MODEL_ERRORS,
    build_analysis_graph,
    build_model,
    default_model_name,
    messages_for_model,
    model_error_message,
    tool_artifacts,
    tool_trace,
)
from src import table_store
from src.table_store import SavedTable, TableStore
from src.tools import DATA_PATH, DEFAULT_PREVIEW_ROWS, AnalysisContext, PreparedDataset
from src.tools.candidates import REASONS, STATUS_LABELS, review_counts
from src.web_journey import describe_journey
from src.web_progress import (
    Activity, activity_for_update, activity_html, describe_call, update_candidate_activity,
    update_prediction_activity, update_validation_activity,
)

DEMO_QUESTIONS = {
    "比较各区转售价": "生成各区平均转售价的柱状图，并指出哪个区最高。",
    "查看转售价分布": "生成转售价分布直方图，并给出中位数和范围。",
    "分析面积与价格": "画一张房屋面积与转售价的散点图，并解释相关性。",
}
CONTEXT_TURNS = 4
VALIDATION_STATUSES = {"passed": "通过", "limited": "部分覆盖", "failed": "未通过"}
SAVE_TABLE_TARGET = r"(?:(?:当前|这个|这份|筛选后(?:的)?|清洗后(?:的)?)\s*)*(?:数据(?:表)?|范围|结果表|筛选结果|清洗结果|筛选版|清洗版)"
SAVE_TABLE_VERSION = r"\s*(?:为|成)\s*(?:一个)?(?:新(?:的)?(?:版本|数据表|表)|筛选版|清洗版)"
SAVE_TABLE_REQUEST = re.compile(
    rf"(?:请\s*)?(?:保存\s*{SAVE_TABLE_TARGET}(?:{SAVE_TABLE_VERSION})?|(?:保存|另存){SAVE_TABLE_VERSION}"
    rf"|(?:把|将)\s*{SAVE_TABLE_TARGET}\s*保存(?:{SAVE_TABLE_VERSION})?)"
    r"|(?:please\s+)?save\s+(?:(?:this|the|current|filtered|cleaned|working)\s+)*(?:data\s+table|dataset|table|selection)(?:\s+as\s+a\s+new\s+(?:table|version))?",
    re.IGNORECASE,
)
SAVE_TABLE_SUFFIX = re.compile(r"[,，。；;]\s*(?:并且?|然后|再)?\s*|(?:\s+and\s+(?:then\s+)?|并且?|然后|再)", re.IGNORECASE)


@dataclass(frozen=True)
class Dataset:
    selected: SavedTable
    original: SavedTable
    frame: pd.DataFrame
    source_frame: pd.DataFrame


@dataclass(frozen=True)
class Report:
    title: str
    filename: str
    image: bytes
    evidence: dict | None = None


@dataclass(frozen=True)
class Turn:
    question: str
    messages: list[AnyMessage] = field(default_factory=list)
    reports: list[Report] = field(default_factory=list)
    trace: dict | None = None
    error: str = ""
    activity: list[Activity] = field(default_factory=list)


def select_dataset(store: TableStore) -> Dataset:
    upload_version = st.session_state.get("upload_widget_version", 0)
    upload = st.file_uploader("上传 CSV", type="csv", key=f"csv_upload_{upload_version}")
    demo = store.save_original(f"{DATA_PATH.name} (synthetic demo)", DATA_PATH.read_bytes(), is_demo=True)
    pending = st.session_state.pop("select_table_next", None)
    fingerprint = (upload.name, sha256(upload.getbuffer()).hexdigest()) if upload is not None else None
    if fingerprint != st.session_state.get("upload_fingerprint"):
        st.session_state.upload_fingerprint = fingerprint
        st.session_state.upload_error = ""
        if upload is not None:
            try:
                pending = store.save_original(upload.name, upload.getvalue()).key
            except (OSError, ValueError, pa.ArrowException):
                st.session_state.upload_error = "无法保存上传的表。请检查 CSV 编码、表头和数据行。"
    if upload_error := st.session_state.get("upload_error"):
        st.error(upload_error)

    entries = store.list()
    keys = sorted(entries, key=lambda key: (key != demo.key, entries[key].name, bool(entries[key].steps), key))
    if pending in entries:
        st.session_state.selected_table = pending
    if st.session_state.get("selected_table") not in entries:
        st.session_state.selected_table = demo.key
    selector, actions = st.columns([5, 1], vertical_alignment="bottom")
    key = selector.selectbox("当前使用的数据表", keys, format_func=lambda key: entries[key].label, key="selected_table")
    selected = entries[key]
    if actions.button("×", help="删除当前选择的数据表", key="delete_selected_table",
                      disabled=key == demo.key, width="stretch"):
        st.session_state.delete_table = (key, store.deletion_versions(selected, entries))
    if confirmation := st.session_state.get("delete_table"):
        if confirmation[0] == key and key != demo.key:
            confirm_table_deletion(store, selected, confirmation[1], demo.key)
        else:
            st.session_state.pop("delete_table")
    if st.session_state.pop("table_deleted", False):
        st.success("已删除数据表。")

    cached = st.session_state.get("loaded_table")
    if cached is not None and cached[0] == key:
        return cached[1]
    original = entries[selected.source_key]
    source_frame = store.read(original)
    frame = source_frame if selected.key == original.key else store.read(selected)
    dataset = Dataset(selected, original, frame, source_frame)
    st.session_state.loaded_table = (key, dataset)
    return dataset


def confirm_table_deletion(store: TableStore, selected: SavedTable, versions: tuple[str, ...], demo_key: str) -> None:
    with st.container(border=True):
        st.warning("永久删除工作台中保存的这张数据表？")
        st.text(selected.label)
        if versions:
            st.caption(f"同时删除基于此表保存的 {len(versions):,} 个筛选版或清洗版。")
        cancel, confirm = st.columns(2)
        if cancel.button("取消", key="cancel_table_deletion", width="stretch"):
            st.session_state.pop("delete_table")
            st.rerun()
        if confirm.button("确认删除", key="confirm_table_deletion", type="primary", width="stretch"):
            st.session_state.pop("delete_table")
            try:
                store.delete(selected.key, versions=versions)
            except ValueError:
                st.error("数据表或其保存的版本已有变化，请重新选择后确认。")
            except OSError:
                st.error("无法完成删除。请检查本地数据文件是否被占用。")
            else:
                if selected.key == selected.source_key:
                    if st.session_state.get("upload_fingerprint") == (selected.name, selected.checksum):
                        st.session_state.upload_widget_version = st.session_state.get("upload_widget_version", 0) + 1
                    st.session_state.select_table_next = demo_key
                else:
                    st.session_state.select_table_next = selected.source_key
                st.session_state.pop("loaded_table", None)
                st.session_state.table_deleted = True
                st.rerun()


def working_table(dataset: Dataset) -> PreparedDataset | None:
    selected = PreparedDataset(dataset.frame, dataset.selected.steps) if dataset.selected.steps else None
    return st.session_state.get("working_table", selected)


def scope_filter_text(conditions) -> str:
    return " · ".join(describe_call("filter_rows", condition).details[0].rstrip("。") for condition in conditions)


def save_working_table(store: TableStore, dataset: Dataset, prepared: PreparedDataset, provider: str) -> SavedTable:
    saved = store.save_prepared(dataset.original, prepared)
    st.session_state.select_table_next = saved.key
    st.session_state.scope = (saved.key, provider)
    st.session_state.pop("working_table", None)
    return saved


def split_save_request(question: str) -> tuple[str, bool]:
    text = question.strip().rstrip("。.!！?？").rstrip()
    if SAVE_TABLE_REQUEST.fullmatch(text):
        return "", True
    for separator in SAVE_TABLE_SUFFIX.finditer(text):
        if SAVE_TABLE_REQUEST.fullmatch(text[separator.end():]):
            return text[:separator.start()].strip(), True
    return question, False


def analyze(
    question: str, dataset: Dataset, store: TableStore, provider: str, model_name: str, api_key: str,
    show_activity: Callable[[list[Activity]], None],
) -> Turn:
    activity: list[Activity] = []
    analysis_question, save_requested = split_save_request(question)
    prepared_before = working_table(dataset)
    chinese = bool(re.search(r"[\u3400-\u9fff]", question))
    try:
        if save_requested and not analysis_question:
            if prepared_before is None or prepared_before.steps == dataset.selected.steps:
                answer = ("当前没有尚未保存的筛选或清洗结果。" if chinese
                          else "There is no unsaved selection or cleaned data in the current scope.")
            else:
                saved = save_working_table(store, dataset, prepared_before, provider)
                answer = (f"已保存「{saved.label}」，可以在侧边栏选择。" if chinese
                          else f"Saved {saved.label}. You can select it in the sidebar.")
            return Turn(question, [HumanMessage(content=question), AIMessage(content=answer)])

        show_activity([Activity("正在查看问题和数据字段")])
        model_name = model_name.strip() or default_model_name(provider)
        model = build_model(provider, model_name, api_key=api_key.strip())
        recent = [turn for turn in st.session_state.turns if turn.messages][-CONTEXT_TURNS:]
        history = [
            message for turn in recent
            for message in messages_for_model(turn.messages, (turn.trace or {}).get("rejected_tool_calls", []))
        ]
        with TemporaryDirectory(prefix="analyst-") as directory:
            context = AnalysisContext(dataset.source_frame, dataset.original.name, Path(directory).resolve())
            graph = build_analysis_graph(model, context)
            selected = PreparedDataset(dataset.frame, dataset.selected.steps) if dataset.selected.steps else None
            inputs = {"messages": [*history, HumanMessage(content=analysis_question)], "prepared": prepared_before,
                      "selected_version": selected}
            result = inputs
            for mode, event in graph.stream(inputs, stream_mode=["updates", "values", "custom"]):
                if mode == "values":
                    result = event
                elif mode == "custom":
                    if (update_candidate_activity(activity, event) or update_prediction_activity(activity, event)
                            or update_validation_activity(activity, event)):
                        show_activity(activity)
                else:
                    for step in activity_for_update(event):
                        if step not in activity:
                            activity.append(step)
                            show_activity(activity)
            messages = result["messages"][len(history):]
            reports = []
            rejected = result.get("rejected_tool_calls", [])
            for artifact in tool_artifacts(messages, rejected):
                if "path" in artifact:
                    path = Path(artifact["path"])
                    reports.append(Report(artifact["title"], path.name, path.read_bytes(), artifact.get("feature_importance")))
        steps = tool_trace(messages, rejected)
        prepared_after = result.get("prepared")
        input_frame = prepared_before.frame if prepared_before is not None else dataset.source_frame
        input_label = dataset.selected.label
        if "working_table" in st.session_state:
            input_label += " · 临时分析范围"
        output_frame = prepared_after.frame if prepared_after is not None else dataset.source_frame
        before_steps = list(prepared_before.steps) if prepared_before is not None else []
        after_steps = list(prepared_after.steps) if prepared_after is not None else []
        temporary = after_steps != list(dataset.selected.steps)
        saved = None
        if result.get("pending_scope_choice") and save_requested:
            answer = messages[-1]
            messages[-1] = answer.model_copy(update={"response_metadata": {**answer.response_metadata, "save_requested": True}})
        if result["request_scope"].get("scope_choice") and history and isinstance(history[-1], AIMessage):
            save_requested = save_requested or history[-1].response_metadata.get("save_requested", False)
        if save_requested and not result.get("pending_scope_choice"):
            validation = result.get("validation", [])
            latest_tools = {step["tool"]: step["status"] for step in steps if not step.get("deferred")}
            ready = bool(steps) and "error" not in latest_tools.values() and (
                not validation or validation[-1]["status"] != "failed"
            ) and not result["request_scope"]["unresolved"]
            if ready and temporary and prepared_after is not None:
                saved = save_working_table(store, dataset, prepared_after, provider)
                status = f"已保存「{saved.label}」。" if chinese else f"Saved {saved.label}."
            else:
                status = "本轮没有可保存的新版本。" if chinese else "This analysis did not produce a new version to save."
            messages[-1] = AIMessage(content=f"{messages[-1].text}\n\n{status}", id=messages[-1].id)
        if saved is None:
            if temporary:
                st.session_state.working_table = prepared_after
            else:
                st.session_state.pop("working_table", None)
        trace = None
        if steps:
            scopes = [
                item["analysis_scope"] for item in tool_artifacts(messages, rejected)
                if "analysis_scope" in item
            ]
            if not scopes and after_steps != before_steps:
                scopes = [{"rows": len(output_frame), "steps": after_steps}]
            analysis_ranges = []
            for scope in scopes:
                from_selected = not dataset.selected.steps or scope["steps"][:len(dataset.selected.steps)] == list(dataset.selected.steps)
                item = {
                    "table": dataset.selected.label if from_selected else dataset.original.label,
                    "rows": scope["rows"],
                    "filters": [step["parameters"] for step in scope["steps"] if step["tool"] == "filter_rows"],
                }
                if item not in analysis_ranges:
                    analysis_ranges.append(item)
            trace = {
                "version": 1,
                "dataset": {
                    "name": dataset.original.name,
                    "sha256": dataset.original.checksum,
                    "rows": len(dataset.source_frame),
                    "columns": {name: str(dtype) for name, dtype in dataset.source_frame.dtypes.items()},
                },
                "question": question,
                "model": {"provider": provider, "name": model_name},
                "input_table": {"label": input_label, "rows": len(input_frame), "columns": len(input_frame.columns)},
                "request_scope": result["request_scope"],
                "analysis_ranges": analysis_ranges,
                "steps": steps,
            }
            if validation := result.get("validation"):
                trace["validation"] = validation
            if rejected:
                trace["rejected_tool_calls"] = rejected
            if evidence := [report.evidence for report in reports if report.evidence is not None]:
                trace["feature_importance"] = evidence
            if before_steps or after_steps or temporary:
                trace["working_data"] = {
                    "before": before_steps,
                    "after": after_steps,
                    "rows_after": len(output_frame),
                    "table_before": dataset.selected.key,
                    "table_after": saved.key if saved else dataset.selected.key,
                    "temporary": temporary and saved is None,
                }
                if saved is not None:
                    trace["working_data"]["saved_version"] = saved.label
        return Turn(question, messages, reports, trace, activity=activity)
    except MODEL_ERRORS as error:
        if isinstance(error, ConnectionError):
            return Turn(question, error="连接不到 Ollama。请启动 Ollama 后重新发送问题。", activity=activity)
        return Turn(question, error=model_error_message(error), activity=activity)
    except ValueError as error:
        return Turn(question, error=f"无法启动分析。{error}", activity=activity)
    except (OSError, pa.ArrowException):
        return Turn(question, error="无法保存数据表或读取报告。请检查本地工作区。", activity=activity)


def show_preview(frame: pd.DataFrame, total_rows: int):
    st.caption(f"显示前 {len(frame):,} 行 · 共 {total_rows:,} 行")
    st.dataframe(frame, hide_index=True, width="stretch")


def show_preparation(dataset: Dataset, store: TableStore):
    entry = dataset.selected
    version_title = "筛选" if entry.filter_only else "清洗"
    with st.expander(f"{version_title}逻辑与代码"):
        st.caption(f"原表 {dataset.original.rows:,} 行 · 当前版本 {entry.rows:,} 行")
        for index, step in enumerate(entry.steps, start=1):
            action = describe_call(step["tool"], step["parameters"])
            st.text(f"{index}. {action.label} · {step['rows_before']:,} → {step['rows_after']:,} 行")
            st.text("\n".join(action.details))
            st.caption(step["detail"])
        script, implementation = store.preparation_code(entry)
        replay_tab, source_tab = st.tabs(["复现脚本", "实际工具代码"])
        with replay_tab:
            st.code(script, language="python")
        with source_tab:
            st.code(implementation, language="python")
        st.download_button(
            f"下载{version_title}版 CSV", dataset.frame.to_csv(index=False).encode("utf-8-sig"),
            entry.export_name, mime="text/csv", key=f"table-csv-{entry.key}", on_click="ignore",
        )


def show_importance_evidence(evidence: dict):
    folds = evidence["folds"]
    with st.expander("因素重要性怎么解读", expanded=True):
        st.write(
            f"用 {evidence['rows']:,} 行做 {folds} 折验证。模型平均绝对误差（MAE）为 "
            f"{evidence['validation_mae']:,.4g}，只猜中位数时为 {evidence['baseline_mae']:,.4g}。"
            f"模型在 {evidence['baseline_wins']}/{folds} 折中误差更低。"
        )
        if evidence["rows"] < evidence.get("available_rows", evidence["rows"]):
            st.caption(f"本次从 {evidence['available_rows']:,} 行有效记录中抽样，抽样种子为 {evidence['sample_seed']}。")
        top = evidence["factors"][0]
        st.text(
            f"打乱「{top['field']}」后，验证误差平均增加 {top['mae_increase']:,.4g}。"
            f"它在 {top['leading_folds']}/{folds} 折中排名第一或并列第一。"
        )
        if top["leading_folds"] < folds:
            st.caption("第一名随验证折变化，具体名次需要更多数据来确认。")
        for factor in evidence["factors"]:
            st.text(f"{factor['field']} · 占比 {factor['share']:.1%} · {factor['positive_folds']}/{folds} 折打乱后误差上升")
        st.caption("占比描述模型在这份数据里的预测依赖。相关字段可能分摊重要性，因果方向和其他数据上的表现需要额外验证。")
        st.caption("占比的分母是逐列打乱后所有正向的误差增量之和。")


def show_candidate_review(review: dict, *, rejected: bool):
    with st.container(border=True):
        method = {"rank_relationships": "关联排名", "feature_importance_report": "因素重要性",
                  "evaluate_prediction": "预测评估", "predict_value": "单条估算"}[review["method"]]
        st.text(f"{method} · 目标字段 {review['target']}")
        counts = review_counts(review)
        status = f"入选 {counts['included']} · 跳过 {counts['excluded']} · 待确认 {counts['needs_confirmation']}"
        st.caption(f"{review['target_rows']:,} 行有有效目标值 · {status}")
        if rejected:
            st.caption("这次计算的结果已被校验节点排除。")
        if review["fields"]:
            kinds = {"numeric": "数值", "categorical": "分类", "date": "日期", "": "—"}
            st.dataframe([
                {"字段": row["field"], "审查结果": STATUS_LABELS[row["status"]], "类型": kinds[row["kind"]],
                 "有效配对行数": row["usable_rows"], "不同取值数": row["distinct_values"],
                 "原因": " ".join(REASONS[code] for code in row["reasons"])}
                for row in review["fields"]
            ], hide_index=True, width="stretch", column_config={"原因": st.column_config.TextColumn(width="large")})
        else:
            st.caption("当前请求没有可审查的候选字段。")
        if counts["needs_confirmation"]:
            first = next(row["field"] for row in review["fields"] if row["status"] == "needs_confirmation")
            st.caption("待确认字段已暂缓纳入。确认其适合本次分析后，可以在聊天框发送下面的指令，重跑上一轮分析。")
            st.code(f"确认纳入 `{first}`", language=None)


def show_execution_record(turn: Turn, trace: dict):
    journey = describe_journey(trace["steps"])
    validation = trace.get("validation", [])
    reviews = [(review, step.get("rejected_by_validation", False)) for step in trace["steps"]
               if (review := step.get("evidence", {}).get("candidate_review"))]
    with st.expander("查看执行记录"):
        labels = ["分析旅程"]
        if reviews:
            labels.append("候选字段审查")
        if validation:
            labels.append("校验结果")
        labels.append("原始工具数据")
        tabs = dict(zip(labels, st.tabs(labels)))
        with tabs["分析旅程"]:
            st.markdown("**本轮问题**")
            st.text(trace["question"])
            input_table = trace.get("input_table")
            if input_table:
                st.text(f"起始数据 · {input_table['label']} · {input_table['columns']:,} 列")
            else:
                st.text(f"数据来源 · {trace['dataset']['name']}")
            if working := trace.get("working_data"):
                st.caption(f"本轮结束时的工作表有 {working['rows_after']:,} 行。")
                if working.get("temporary"):
                    st.caption("本轮使用临时分析范围，未创建数据版本。")
                if saved := working.get("saved_version"):
                    st.caption(f"已保存 {saved}。")
            if replaced := trace.get("request_scope", {}).get("replaced_conditions"):
                st.caption("本轮移除或替换的临时筛选")
                for condition in replaced:
                    st.text(describe_call("filter_rows", condition).details[0])
            if confirmed := trace.get("request_scope", {}).get("clarified_question"):
                st.caption("确认范围前的问题")
                st.text(confirmed)
            for number, step in enumerate(journey, start=1):
                with st.container(border=True):
                    st.markdown(f"**{number}. {step['action']}**")
                    round_label = f"第 {step['round']} 轮 · " if step["round"] is not None else ""
                    st.caption(f"{round_label}{step['status']}")
                    if trace["steps"][number - 1].get("rejected_by_validation"):
                        st.caption("校验未通过，结果已排除。")
                    if step["purpose"]:
                        st.text(f"用途 · {step['purpose']}")
                    for detail in step["method"]:
                        st.text(f"做法 · {detail}")
                    for note in step["outcome"]:
                        st.text(note)
                    for table in step["tables"]:
                        st.caption(table["title"])
                        st.dataframe(table["rows"], hide_index=True, width="stretch")
        if reviews:
            with tabs["候选字段审查"]:
                for review, rejected in reviews:
                    show_candidate_review(review, rejected=rejected)
        if validation:
            with tabs["校验结果"]:
                for attempt in validation:
                    with st.container(border=True):
                        status = VALIDATION_STATUSES[attempt["status"]]
                        st.markdown(f"**第 {attempt['attempt']} 次校验 · {status}**")
                        for item in attempt["checks"]:
                            tool = describe_call(item["tool"], {}) if item["tool"] else None
                            method = f"{tool.label} · " if tool else ""
                            st.text(f"{method}{item['label']} · {VALIDATION_STATUSES[item['status']]}")
                            st.caption(item["detail"])
                            if scope := item.get("scope_comparison"):
                                for label, key, empty in (
                                    ("请求的筛选", "requested", "本轮未识别到可核对的新增条件"),
                                    ("实际生效的筛选", "actual", "无"),
                                ):
                                    rules = [
                                        describe_call("filter_rows", condition).details[0]
                                        + (f"（匹配原话「{condition['phrase']}」）" if condition.get("source") == "model_label_match" else "")
                                        for condition in scope[key]
                                    ]
                                    st.text(f"{label}\n" + ("\n".join(f"• {rule}" for rule in rules) or empty))
                        if attempt["retrying"]:
                            st.caption("已根据这次校验的发现发起重试。")
        with tabs["原始工具数据"]:
            for number, step in enumerate(trace["steps"], start=1):
                st.text(f"{number}. {step['tool']} · {journey[number - 1]['status']}")
                st.json(step["parameters"], expanded=False)
                st.code(step["result"], language=None, wrap_lines=True)
        st.download_button(
            "下载执行记录 JSON", json.dumps({**trace, "journey": journey}, ensure_ascii=False, indent=2),
            "analysis-trace.json", mime="application/json",
            key=f"trace-{turn.messages[0].id}", on_click="ignore",
        )


def show_turn(turn: Turn):
    with st.chat_message("user", avatar=":material/person:"):
        st.text(turn.question)
    with st.chat_message("assistant", avatar=":material/analytics:"):
        activity = getattr(turn, "activity", [])
        if activity:
            st.html(activity_html(activity, running=False))
        if turn.error:
            st.error(turn.error)
            return
        trace = getattr(turn, "trace", None)
        if trace:
            if expanded := trace.get("request_scope", {}).get("expanded_for_comparison"):
                st.caption(f"本题要比较「{expanded['column']}」。已放宽上轮临时筛选，可比较的分组从 "
                           f"{expanded['kept_groups']:,} 个增至 {expanded['wide_groups']:,} 个。")
            for scope in trace.get("analysis_ranges", []):
                st.caption(f"本轮数据 · {scope['table']} · 使用范围 {scope['rows']:,} 行")
                if filters := scope_filter_text(scope["filters"]):
                    st.caption(filters)
        st.markdown(turn.messages[-1].text.strip() or "模型没有返回可显示的回复。")
        if trace and (validation := trace.get("validation")):
            last = validation[-1]["status"]
            label, render = {
                "passed": ("校验通过", st.success),
                "limited": ("部分校验 · 查看已核对的内容和复算范围", st.warning),
                "failed": ("校验未通过", st.error),
            }[last]
            retry_note = " · 已重试一次" if len(validation) > 1 else ""
            render(f"{label}{retry_note}")
            if last == "passed":
                st.caption("已核对可识别的筛选范围、受支持的统计量和常见结论措辞。")
            elif last == "limited" and any(check.get("scope_comparison", {}).get("status") == "limited"
                                          for check in validation[-1]["checks"]):
                st.caption("筛选范围只核对了一部分。打开「校验结果」查看哪些条件尚待确认。")
        for artifact in tool_artifacts(turn.messages, (trace or {}).get("rejected_tool_calls", [])):
            if "table" in artifact:
                show_preview(pd.DataFrame(**artifact["table"]), artifact["total_rows"])
            if groups := artifact.get("group_summary"):
                st.dataframe(pd.DataFrame(groups["groups"]).rename(columns={
                    "group": "分组", "value": "统计值", "valid_rows": "参与统计的有效行数",
                }), hide_index=True, width="stretch")
            if ranked := artifact.get("ranked_rows"):
                st.dataframe(pd.DataFrame(ranked["rows"], columns=ranked["columns"]), hide_index=True, width="stretch")
        for report in turn.reports:
            with st.container(border=True):
                with st.container(key=f"report-image-{Path(report.filename).stem}"):
                    st.image(report.image, caption=report.title, width="stretch")
                evidence = getattr(report, "evidence", None)
                if evidence:
                    show_importance_evidence(evidence)
                st.download_button(
                    "下载 PNG", report.image, report.filename, mime="image/png",
                    key=report.filename, on_click="ignore",
                )
        if trace:
            show_execution_record(turn, trace)


def main():
    st.set_page_config(page_title="数据分析工作台", page_icon=":material/analytics:", layout="wide")
    st.html(Path(__file__).with_name("web.css"))

    store = TableStore(table_store.DATASET_DIRECTORY)
    with st.sidebar:
        st.subheader("数据源", anchor=False)
        try:
            dataset = select_dataset(store)
        except (OSError, ValueError, pa.ArrowException):
            st.error("无法读取本地数据表。请检查工作区中的数据文件。")
            st.stop()
        st.caption(f"{len(dataset.frame):,} 行 · {len(dataset.frame.columns):,} 列")
        with st.expander("查看字段"):
            for column, dtype in dataset.frame.dtypes.items():
                st.text(f"{column} · {dtype}")

        st.divider()
        st.subheader("分析模型", anchor=False)
        provider = st.radio(
            "模型服务", ("ollama", "openai"),
            index=int(os.getenv("MODEL_PROVIDER") == "openai"),
            format_func={"ollama": "本地 Ollama", "openai": "OpenAI API"}.get,
        )
        model_name = st.text_input("模型名称", value=default_model_name(provider), key=f"model_{provider}")
        api_key = ""
        if provider == "openai":
            api_key = st.text_input(
                "OpenAI API Key", type="password",
                help="留空时读取项目环境中的 OPENAI_API_KEY。",
            )
            st.caption("问题、字段信息和工具结果会发送给 OpenAI。")

        scope = (dataset.selected.key, provider)
        if st.session_state.get("scope") != scope:
            st.session_state.scope = scope
            st.session_state.turns = []
            st.session_state.pop("working_table", None)
        st.divider()
        if st.button("新对话", width="stretch", disabled=not st.session_state.turns):
            st.session_state.turns = []
            st.session_state.pop("working_table", None)
        prepared = working_table(dataset)
        frame = prepared.frame if prepared is not None else dataset.source_frame
        if "working_table" in st.session_state:
            st.caption(f"当前对话使用临时分析范围 · {len(frame):,} 行")
            if prepared is not None:
                active_filters = [step["parameters"] for step in prepared.steps if step["tool"] == "filter_rows"]
                if filters := scope_filter_text(active_filters):
                    st.caption(filters)
            if prepared is not None and st.button("保存当前范围为数据版本", width="stretch"):
                try:
                    save_working_table(store, dataset, prepared, provider)
                except (OSError, ValueError, pa.ArrowException):
                    st.error("无法保存当前范围。请检查本地工作区。")
                else:
                    st.rerun()
            if st.button("清除临时范围", width="stretch"):
                st.session_state.pop("working_table", None)
                st.rerun()
        if dataset.selected.steps and st.button("恢复原始数据", width="stretch"):
            st.session_state.select_table_next = dataset.original.key
            st.session_state.scope = (dataset.original.key, provider)
            st.session_state.pop("working_table", None)
            st.rerun()

    st.title("数据分析工作台", anchor=False)
    st.markdown("用自然语言分析 CSV，生成可下载的图表。")
    with st.expander("预览数据"):
        if "working_table" in st.session_state:
            st.caption("预览当前对话的临时分析范围。")
        show_preview(frame.head(DEFAULT_PREVIEW_ROWS), len(frame))
    if dataset.selected.steps:
        show_preparation(dataset, store)

    example = None
    if not st.session_state.turns:
        with st.container(border=True, key="start_here"):
            st.subheader("开始分析", anchor=False)
            st.markdown("选择一个示例，或在下方输入自己的问题。")
            prompts = DEMO_QUESTIONS if dataset.selected.is_demo and not dataset.selected.steps else {
                "查看数据行": "我可以看一下这份数据吗？",
                "检查数据质量": "请检查这份数据的字段类型、缺失值和不同取值数。",
            }
            for area, (label, question) in zip(st.columns(len(prompts)), prompts.items()):
                if area.button(label, width="stretch"):
                    example = question

    for turn in st.session_state.turns:
        show_turn(turn)

    if st.session_state.turns:
        latest = st.session_state.turns[-1].messages
        if latest and isinstance(latest[-1], AIMessage):
            pending = latest[-1].response_metadata.get("scope_choice")
            if pending and pending["steps"] == list(prepared.steps if prepared is not None else ()):
                choices = ("沿用刚才的筛选", "按所选表的全部范围")
                for area, label in zip(st.columns(len(choices)), choices):
                    if area.button(label, width="stretch"):
                        example = label

    question = st.chat_input("输入你的分析问题…") or example
    if question and question.strip():
        question = question.strip()
        with st.chat_message("user", avatar=":material/person:"):
            st.text(question)
        with st.chat_message("assistant", avatar=":material/analytics:"):
            progress_slot = st.empty()

            def show_activity(steps):
                progress_slot.html(activity_html(steps, running=True))

            turn = analyze(question, dataset, store, provider, model_name, api_key, show_activity)
        st.session_state.turns.append(turn)
        st.rerun()


if __name__ == "__main__":
    main()
