"""Review fields for the requested analysis without changing the working table."""

import re
from dataclasses import dataclass

import pandas as pd
from langchain.messages import HumanMessage
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .dataset import finite_numbers

CANDIDATE_TOOLS = {"rank_relationships", "feature_importance_report", "evaluate_prediction", "predict_value"}
MIN_PAIRED_ROWS = 3
CONFIRMATION = re.compile(r"^[ \t]*(?:确认纳入|confirm inclusion of)[ \t]*((?:`[^`\n]+`[ \t,，、]*)+)$", re.I | re.M)
COLUMN_TOKENS = re.compile(r"[^\w]+|_", re.UNICODE)
ID_NAME = re.compile(r"(?:编号|标识符|订单号|工单号|单据号|流水号|记录号|客户号|账号|序号)$")
ID_TOKENS = {"id", "uuid", "guid", "identifier", "sku"}
ID_NUMBERS = {"row", "serial", "record", "order", "customer", "account", "invoice", "transaction"}
AFTER_OUTCOME = re.compile(
    r"^(?:post|after)_(?:sale|order|payment|transaction|checkout|delivery|outcome)(?:_|$)"
    r"|(?:成交后|结算后|付款后|交付后|结果公布后)"
)

REASONS = {
    "unknown_field": "当前表中找不到这个字段。",
    "date": "日期列需要先提取有意义的时间特征。",
    "few_pairs": "和目标列同时有值的记录少于 3 行。",
    "constant": "有效记录中的取值没有变化。",
    "constant_target": "这些记录里的目标值没有变化。",
    "small_groups": "分类相关比 η 至少需要两组，每组至少有 2 条有效记录。",
    "unique_labels": "分类取值都只出现了一次，难以在验证数据中复用。",
    "identifier": "列名像编号或标识符，请确认其数值或分类有可解释的含义。",
    "target_copy": "在两列同时有效的记录中，取值与目标完全一致，可能是目标的副本。",
    "known_after": "列名提示该字段可能在结果发生后才产生，请确认预测时能否取得。",
    "many_missing": "至少一半有目标值的记录缺少该字段，解读时留意可用样本。",
    "manual_confirmation": "用户在本轮明确确认纳入。",
    "eligible": "满足当前方法的基本样本及变化要求。",
    "uncomputable": "该字段的统计量未能计算为有限值。",
}
STATUS_LABELS = {"included": "入选", "excluded": "跳过", "needs_confirmation": "待确认"}


@dataclass
class CandidateReview:
    evidence: dict
    values: dict[str, pd.Series]

    def skip_uncomputable(self, name: str) -> None:
        row = next(row for row in self.evidence["fields"] if row["field"] == name)
        row["status"] = "excluded"
        row["reasons"] = ["uncomputable", *row["reasons"]]


def confirmed_fields(messages) -> set[str]:
    question = next((message.text for message in reversed(messages) if isinstance(message, HumanMessage)), "")
    return {name for match in CONFIRMATION.finditer(question) for name in re.findall(r"`([^`]+)`", match[1])}


def field_tokens(name: str) -> list[str]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return [part for part in COLUMN_TOKENS.split(separated.casefold()) if part]


def review_candidates(
    frame: pd.DataFrame, target_column: str, target: pd.Series, candidate_columns: list[str] | None,
    *, method: str, confirmed: set[str],
) -> CandidateReview:
    prediction = method != "rank_relationships"
    names = dict.fromkeys(frame.columns if candidate_columns is None else candidate_columns)
    names.pop(target_column, None)
    target = target.dropna()
    review = CandidateReview({"method": method, "target": target_column, "target_rows": len(target), "fields": []}, {})
    aligned = frame.loc[target.index, [name for name in names if name in frame.columns]]
    for name in names:
        record = {"field": name, "kind": "", "status": "excluded", "usable_rows": 0, "distinct_values": 0, "reasons": []}
        review.evidence["fields"].append(record)
        if name not in aligned:
            record["reasons"] = ["unknown_field"]
            continue

        series = aligned[name]
        numeric = is_numeric_dtype(series) and not is_bool_dtype(series)
        date = is_datetime64_any_dtype(series)
        record["kind"] = "date" if date else "numeric" if numeric else "categorical"
        if numeric:
            series = finite_numbers(series)
        elif not date and not is_bool_dtype(series):
            labels = series.astype("string")
            if prediction:
                labels = labels.str.strip()
            series = labels.mask(labels.str.strip().eq(""))
        valid = series.dropna()
        distinct = valid.nunique()
        record.update(usable_rows=len(valid), distinct_values=distinct)

        if date:
            reason = "date"
        elif len(valid) < MIN_PAIRED_ROWS:
            reason = "few_pairs"
        elif distinct < 2:
            reason = "constant"
        elif not prediction and target.loc[valid.index].nunique() < 2:
            reason = "constant_target"
        elif not numeric and prediction and valid.value_counts().max() < 2:
            reason = "unique_labels"
        elif not numeric and not prediction and valid.value_counts().min() < 2:
            reason = "small_groups"
        else:
            reason = ""

        tokens = field_tokens(name)
        risks = []
        numbered_identifier = len(tokens) == 2 and tokens[0] in ID_NUMBERS and tokens[1] in {"number", "no"}
        if ID_TOKENS.intersection(tokens) or numbered_identifier or ID_NAME.search(name):
            risks.append("identifier")
        if numeric and len(valid) >= MIN_PAIRED_ROWS and distinct > 1 and valid.eq(target.loc[valid.index]).all():
            risks.append("target_copy")
        if prediction and AFTER_OUTCOME.search("_".join(tokens)):
            risks.append("known_after")

        if reason:
            record["reasons"] = [reason, *risks]
        elif risks and name not in confirmed:
            record.update(status="needs_confirmation", reasons=risks)
        else:
            record.update(status="included", reasons=[*risks, "manual_confirmation"] if risks else ["eligible"])
            review.values[name] = series
        if len(target) and 2 * len(valid) <= len(target):
            record["reasons"].append("many_missing")
    return review


def candidates_for_tool(runtime, frame, target_column, target, candidate_columns, method) -> CandidateReview:
    if (review := runtime.state.get("candidate_review")) is not None:
        return review
    return review_candidates(
        frame, target_column, target, candidate_columns, method=method,
        confirmed=confirmed_fields(runtime.state.get("messages", [])),
    )


def review_counts(review: dict) -> dict[str, int]:
    return {status: sum(row["status"] == status for row in review["fields"]) for status in STATUS_LABELS}


def review_summary(review: dict) -> str:
    counts = review_counts(review)
    lines = [f"Candidate review: {counts['included']} included, {counts['excluded']} skipped, "
             f"{counts['needs_confirmation']} pending user confirmation."]
    for heading, status in (("Skipped", "excluded"), ("Pending confirmation (excluded from this calculation)", "needs_confirmation")):
        fields = [f"{row['field']} ({', '.join(row['reasons'])})" for row in review["fields"] if row["status"] == status]
        if fields:
            lines.append(f"{heading}: {', '.join(fields[:10])}" + (f" and {len(fields) - 10} more." if len(fields) > 10 else "."))
    if counts["needs_confirmation"]:
        lines.append("For confirmation, the user can send: 确认纳入 `field_name` (English: Confirm inclusion of `field_name`).")
    return "\n".join(lines)
