"""Translate LangGraph updates into short, visible analysis activity messages."""

from dataclasses import dataclass
from html import escape

from langchain.messages import AIMessage

from src.agent import first_preparation_call, tool_parameters
from src.tools.candidates import CANDIDATE_TOOLS


@dataclass(frozen=True)
class Activity:
    label: str
    phase: str = "review"
    details: tuple[str, ...] = ()
    tool_call_id: str = ""


METRIC_LABELS = {"mean": "平均值", "median": "中位数", "sum": "总和", "min": "最小值", "max": "最大值"}
PERIOD_LABELS = {"day": "天", "week": "周", "month": "月", "quarter": "季度", "year": "年"}
PREDICTION_MODELS = {"decision_tree": "单棵决策树", "linear_regression": "线性回归", "ridge": "岭回归", "random_forest": "随机森林"}
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
        case "count_rows":
            return Activity("统计记录数", details=("统计当前工作表的所有行，包含其他字段有缺失值的记录。",))
        case "aggregate_column":
            metric = METRIC_LABELS.get(params.get("statistic"), "统计值")
            return Activity("计算整列统计", details=(f"在当前数据范围计算{field_label(value)}的{metric}。",))
        case "formula_group_totals":
            return Activity("逐行计算公式并分组求和", details=(
                f"按{field_label(params.get('group_column'))}汇总{field_label(value)}。公式为 {params.get('expression', '')}。",
            ))
        case "rank_rows":
            field = field_label(params.get("sort_column"))
            return Activity("排序并列出原始记录", details=(f"按{field}排序，保留指定的字段和记录条数。",))
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
        case "group_distribution_report":
            return Activity("绘制分组箱线图", "chart", (f"按{field_label(params.get('group_column'))}展示{field_label(value)}的分布。",))
        case "rank_relationships":
            target = field_label(params.get("target_column"))
            candidates = params.get("candidate_columns")
            fields = f"入选的 {len(candidates)} 个字段" if candidates is not None else "符合条件的字段"
            detail = f"比较{fields}与{target}的关联。数值列按 |Pearson r| 排名，分类列单独计算 η。"
            return Activity("比较字段关联强度", details=(detail,))
        case "feature_importance_report":
            target = field_label(params.get("target_column"))
            detail = f"预测{target}，在验证数据上逐列打乱取值，比较误差增量并绘制前 {params.get('top_n')} 个因素的占比。"
            return Activity("绘制因素重要性图", "chart", (detail,))
        case "evaluate_prediction" | "predict_value":
            target = field_label(params.get("target_column"))
            title = "估算单条记录" if name == "predict_value" else "评估预测误差"
            return Activity(title, details=(f"用留出的记录验证{target}的预测误差，并与中位数基准比较。",))
        case "relationship_report":
            x_column = field_label(params.get("x_column"))
            y_column = field_label(params.get("y_column"))
            detail = f"横轴用{x_column}，纵轴用{y_column}，并计算 Pearson 相关系数。"
            return Activity("绘制散点图", "chart", (detail,))
    return None


def activity_for_update(update):
    for node, state in update.items():
        if node not in {"analyst", "request_preview", "chart_ranking", "execute_row_step"}:
            continue
        for message in (state or {}).get("messages", []):
            if not isinstance(message, AIMessage):
                continue
            preparation = first_preparation_call(message.tool_calls)
            calls = [preparation] if preparation is not None else message.tool_calls
            actions = tuple(dict.fromkeys(
                action for call in calls
                if call["name"] not in CANDIDATE_TOOLS and (action := describe_call(call["name"], call["args"])) is not None
            ))
            if len(actions) == 1:
                yield actions[0]
            elif actions:
                phase = "chart" if any(action.phase == "chart" for action in actions) else "review"
                details = tuple(f"{action.label}。{action.details[0]}" for action in actions)
                yield Activity(f"同时执行 {len(actions)} 项分析", phase, details)


def update_candidate_activity(activity: list[Activity], event: dict) -> bool:
    if event.get("event") == "candidate_calculation":
        activity.append(describe_call(event["tool"], event["args"]))
        return True
    if event.get("event") != "candidate_review":
        return False
    target = field_label(event["target"])
    label = "正在审查候选字段"
    scope = f"为目标{target}检查有效样本、取值变化和字段用途。"
    if event["stage"] == "start":
        activity.append(Activity(label, details=(scope,)))
    else:
        counts = event["counts"]
        summary = f"入选 {counts['included']} 个，跳过 {counts['excluded']} 个，待确认 {counts['needs_confirmation']} 个。"
        for index in range(len(activity) - 1, -1, -1):
            if activity[index] == Activity(label, details=(scope,)):
                activity[index] = Activity(label, details=(scope, summary))
                break
    return True


def update_prediction_activity(activity: list[Activity], event: dict) -> bool:
    if event.get("event") != "prediction_progress":
        return False
    target = field_label(event["target"])
    phase = "review"
    match event["stage"]:
        case "fit":
            label = "正在训练并验证"
            method = {"random": "随机训练/测试切分", "chronological": "按时间训练/测试切分"}.get(event.get("split"))
            step = method or f"第 {event['fold']}/{event['folds']} 折"
            model = PREDICTION_MODELS.get(event.get("model"))
            model_detail = f"使用{model}，" if model else ""
            detail = f"{step}。{model_detail}用留出记录评估{target}的预测误差。"
            if event.get("sampled", event["rows"] < event["available_rows"]):
                detail += f"本次从 {event['available_rows']:,} 行有效记录中抽取 {event['rows']:,} 行。"
        case "importance":
            label = "正在计算因素重要性"
            detail = f"第 {event['fold']}/{event['folds']} 折。逐列打乱验证数据，比较误差变化。"
        case "chart":
            label = "正在绘制因素重要性图"
            detail = "根据已计算的误差变化绘制占比图。"
            phase = "chart"
        case "refit":
            label = "正在拟合估算模型"
            detail = f"交叉验证已完成，用选定样本拟合估算模型以预测{target}。"
        case _:
            return False
    step = Activity(label, phase, (detail,), event["tool_call_id"])
    for index in range(len(activity) - 1, -1, -1):
        item = activity[index]
        if item.tool_call_id == step.tool_call_id and item.label == label:
            activity[index] = step
            break
    else:
        activity.append(step)
    return True


VALIDATION_ACTIVITIES = {
    "scope": Activity("正在校验", "validation", ("按清洗与筛选步骤重建工具使用的数据。",)),
    "numbers": Activity("正在校验", "validation", ("独立复算已覆盖的统计量，并与工具结果对照。",)),
    "conclusion": Activity("正在校验", "validation", ("检查结论里是否有超出数据证据的表述。",)),
    "retry": Activity("校验发现问题，正在纠正", "validation", ("根据校验记录纠正数据、计算或结论，再校验一次。",)),
}


def update_validation_activity(activity: list[Activity], event: dict) -> bool:
    if event.get("event") != "validation" or (step := VALIDATION_ACTIVITIES.get(event.get("stage"))) is None:
        return False
    if step.label == "正在校验" and activity and activity[-1].label == step.label:
        if event["stage"] == "numbers":
            details = (*VALIDATION_ACTIVITIES["scope"].details, *step.details)
        elif event["stage"] == "conclusion":
            details = (*activity[-1].details, *step.details)
        else:
            details = step.details
        activity[-1] = Activity(step.label, step.phase, details)
    else:
        activity.append(step)
    return True


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
