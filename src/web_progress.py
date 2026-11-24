"""Translate LangGraph updates into short, visible analysis activity messages."""

from dataclasses import dataclass
from html import escape

from langchain.messages import AIMessage

from src.agent import first_preparation_call, tool_parameters


@dataclass(frozen=True)
class Activity:
    label: str
    phase: str = "review"
    details: tuple[str, ...] = ()


METRIC_LABELS = {"mean": "平均值", "median": "中位数", "sum": "总和"}
PERIOD_LABELS = {"day": "天", "week": "周", "month": "月", "quarter": "季度", "year": "年"}
FILTER_LABELS = {
    "equals": "等于", "not_equals": "不等于", "in": "属于", "contains": "包含",
    "greater_than": "大于", "at_least": "至少为", "less_than": "小于", "at_most": "至多为",
}


def field_label(name) -> str:
    return f"「{name}」" if isinstance(name, str) and name.strip() else "待选字段"


def describe_call(name: str, args: dict) -> Activity | None:
    params = tool_parameters(name, args)
    value = params.get("value_column")
    aggregation = params.get("aggregation", "mean")
    statistic = METRIC_LABELS.get(aggregation) if isinstance(aggregation, str) else None
    metric = f"{field_label(value)}的{statistic}" if statistic else "指定指标"
    if aggregation == "count":
        metric = "记录数"

    match name:
        case "dataset_overview":
            return Activity("查看数据结构", details=("读取列名、数据类型和总行数。",))
        case "dataset_preview":
            limit = params.get("limit")
            row_range = f"最多 {limit} 行" if isinstance(limit, int) and 1 <= limit <= 50 else "记录"
            detail = f"读取表格开头{row_range}，保留原始列。"
            return Activity("预览数据", details=(detail,))
        case "profile_dataset":
            return Activity("检查数据质量", details=("检查选定列的数据类型、空值和不同取值数。",))
        case "inspect_column_values":
            field = field_label(params.get("column_name"))
            return Activity("查看字段取值", details=(f"检查{field}的常见值和各值的行数。",))
        case "filter_rows":
            field = field_label(params.get("column_name"))
            operator = params.get("operator")
            comparison = FILTER_LABELS.get(operator, "匹配") if isinstance(operator, str) else "匹配"
            return Activity("筛选数据", details=(f"只保留{field}{comparison} {params.get('value')} 的行。",))
        case "drop_missing" | "drop_duplicates":
            columns = params.get("columns")
            fields = "、".join(map(field_label, columns)) if isinstance(columns, list) and columns else "选定字段"
            if name == "drop_missing":
                return Activity("清理缺失值", details=(f"删除在{fields}中有空值的行。",))
            fields = fields if columns else "整行"
            return Activity("去除重复行", details=(f"按{fields}去重，保留每组首次出现的行。",))
        case "convert_column":
            field = field_label(params.get("column_name"))
            data_type = params.get("data_type")
            labels = {"number": "数值", "date": "日期", "text": "文本"}
            label = labels.get(data_type, "指定类型") if isinstance(data_type, str) else "指定类型"
            return Activity("转换字段类型", details=(f"将{field}转为{label}，统计无法转换的值。",))
        case "reset_dataset":
            return Activity("恢复原始数据", details=("清除当前筛选和转换，重新使用原始 CSV。",))
        case "average_by_group" | "group_report":
            group_column = field_label(params.get("group_column"))
            top_n = params.get("top_n")
            detail = f"按{group_column}分组，计算{metric}"
            if isinstance(top_n, int) and 1 <= top_n <= 25:
                detail += f"，按结果降序取前 {top_n} 组"
            detail += "。"
            if name == "group_report":
                return Activity("绘制分组柱状图", "chart", (detail,))
            return Activity("比较分组均值", details=(detail,))
        case "trend_report":
            date_column, frequency = field_label(params.get("date_column")), params.get("frequency")
            period = PERIOD_LABELS.get(frequency, "指定时间段") if isinstance(frequency, str) else "指定时间段"
            detail = f"以{date_column}为时间列，按{period}计算{metric}。"
            return Activity("绘制趋势折线图", "chart", (detail,))
        case "distribution_report":
            bins = params.get("bins")
            bin_count = f" {bins} 个" if isinstance(bins, int) and 1 <= bins <= 50 else "多个"
            detail = f"把{field_label(value)}分成{bin_count}等宽区间，统计每个区间的行数。"
            return Activity("绘制分布直方图", "chart", (detail,))
        case "rank_relationships":
            target = field_label(params.get("target_column"))
            candidates = params.get("candidate_columns")
            fields = "、".join(map(field_label, candidates)) if candidates else "其余字段"
            detail = f"逐列比较{fields}与{target}的关联。数值列按 |Pearson r| 排名，分类列单独计算 η。"
            return Activity("比较字段关联强度", details=(detail,))
        case "relationship_report":
            x_column = field_label(params.get("x_column"))
            y_column = field_label(params.get("y_column"))
            detail = f"横轴用{x_column}，纵轴用{y_column}，并计算 Pearson 相关系数。"
            return Activity("绘制散点图", "chart", (detail,))
    return None


def activity_for_update(update):
    for node, state in update.items():
        if node not in {"analyst", "request_preview"}:
            continue
        for message in (state or {}).get("messages", []):
            if not isinstance(message, AIMessage):
                continue
            preparation = first_preparation_call(message.tool_calls)
            calls = [preparation] if preparation is not None else message.tool_calls
            actions = tuple(dict.fromkeys(
                action for call in calls
                if (action := describe_call(call["name"], call["args"])) is not None
            ))
            if len(actions) == 1:
                yield actions[0]
            elif actions:
                phase = "chart" if any(action.phase == "chart" for action in actions) else "review"
                details = tuple(f"{action.label}。{action.details[0]}" for action in actions)
                yield Activity(f"同时执行 {len(actions)} 项分析", phase, details)


WORKSTATION = """
<div class="analyst-workstation" aria-hidden="true">
    <span class="desk-shadow"></span>
    <span class="desk-monitor">
        <span class="desk-screen">
            <i class="screen-grid"></i>
            <span class="screen-bars"><i></i><i></i><i></i><i></i></span>
        </span>
    </span>
    <span class="desk-surface"></span>
    <span class="desk-keyboard"></span>
    <span class="analyst-sleeve sleeve-left"></span>
    <span class="analyst-sleeve sleeve-right"></span>
    <span class="analyst-hand hand-left"></span>
    <span class="analyst-hand hand-right"></span>
    <span class="analyst-shirt"></span>
    <span class="analyst-head">
        <i class="analyst-neck"></i><i class="analyst-hair"></i>
    </span>
    <span class="analyst-chair"></span>
</div>
"""


def activity_html(steps: list[Activity], *, running: bool) -> str:
    def bubble_text(step: Activity) -> str:
        details = "".join(
            f'<span class="activity-detail">{escape(detail)}</span>'
            for detail in getattr(step, "details", ())
        )
        return f'<strong class="activity-label">{escape(step.label)}</strong>{details}'

    archived = steps[:-1] if running else steps
    history = "".join(
        f'<li class="activity-bubble activity-past">{bubble_text(step)}</li>' for step in archived
    )
    if history:
        history = f'<ol class="activity-history">{history}</ol>'

    scene = ""
    state = "settled"
    if running:
        step = steps[-1]
        state = f"running phase-{escape(step.phase)}"
        scene = (
            f'<div class="activity-scene">{WORKSTATION}'
            f'<div class="activity-bubble activity-current" role="status">{bubble_text(step)}</div>'
            '</div>'
        )
    return f'<div class="agent-activity {state}" aria-label="分析进度">{history}{scene}</div>'
