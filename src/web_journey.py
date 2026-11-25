"""Readable descriptions of recorded tool decisions and results."""

from src.web_progress import describe_call
from src.tools.candidates import review_counts

PURPOSES = {
    "dataset_overview": "先确认列名和数据规模，再选择适合的计算方法。",
    "dataset_preview": "抽看数据行可以确认字段含义和实际录入格式。",
    "profile_dataset": "缺失值、字段类型和取值数量会影响后续能做哪些统计。",
    "inspect_column_values": "先确认表里真实使用的标签，筛选时才能匹配正确的记录。",
    "filter_rows": "先确定这次要分析的范围，后续统计才会用到符合条件的记录。",
    "drop_missing": "让后续分析使用指定字段都有值的记录。",
    "drop_duplicates": "按指定的去重标准，避免相同记录被重复计入。",
    "convert_column": "让字段能按数值、日期或文本的规则参与后续操作。",
    "reset_dataset": "后续分析可以重新从上传时的完整表开始。",
    "average_by_group": "把同一指标按类别分别汇总，便于比较各组。",
    "group_report": "把各组的统计结果放在同一尺度上比较。",
    "trend_report": "按相同时间间隔汇总，便于观察指标随时间的变化。",
    "distribution_report": "分区间数记录，可以看出值主要集中在哪里。",
    "rank_relationships": "回答哪个字段关联更强，需要把多个候选字段逐一和同一个目标列比较。",
    "relationship_report": "同时查看两个字段怎样变化，并用相关系数量化它们的线性关联。",
    "feature_importance_report": "比较验证时打乱各个字段带来的误差变化，可以衡量模型更依赖哪些字段做预测。",
}


def summarize_evidence(evidence: dict) -> tuple[list[str], list[dict]]:
    notes, tables = [], []
    if changes := evidence.get("row_change"):
        notes.append(f"这一步处理前有 {changes['before']:,} 行，处理后有 {changes['after']:,} 行。")
    if overview := evidence.get("overview"):
        notes.append(f"当前表有 {overview['rows']:,} 行、{overview['columns']:,} 列。")
    if preview := evidence.get("preview"):
        notes.append(f"预览显示前 {preview['rows']:,} 行，当前表共 {preview['total_rows']:,} 行。")
    if profile := evidence.get("profile"):
        notes.append(f"在 {profile['rows']:,} 行中检查了 {len(profile['fields']):,} 个字段。")
        tables.append({"title": "字段检查", "rows": [
            {"字段": row["field"], "类型": row["dtype"], "空值数": row["missing"], "不同取值数": row["distinct"]}
            for row in profile["fields"]
        ]})
    if values := evidence.get("column_values"):
        notes.append(f"找到 {values['distinct']:,} 个不同的非空取值，另有 {values['missing']:,} 行缺失。")
        if values["values"]:
            tables.append({"title": "常见取值", "rows": [
                {"取值": row["label"], "行数": row["rows"]} for row in values["values"]
            ]})
    if ranking := evidence.get("relationship_ranking"):
        notes.append(f"在 {ranking['rows']:,} 行中检查了 {ranking['candidate_count']:,} 个候选字段。")
        if ranking["numeric"]:
            tables.append({"title": "数值字段 · 按 Pearson r 的绝对值排序", "rows": [
                {"字段": row["field"], "Pearson r": f"{row['r']:+.3f}", "有效配对行数": row["rows"]}
                for row in ranking["numeric"]
            ]})
        if ranking["categorical"]:
            tables.append({"title": "分类字段 · 按相关比 η 排序", "rows": [
                {"字段": row["field"], "相关比 η": f"{row['eta']:.3f}", "有效行数": row["rows"], "类别数": row["groups"]}
                for row in ranking["categorical"]
            ]})
        if ranking["skipped_count"] and "candidate_review" not in evidence:
            notes.append(f"有 {ranking['skipped_count']:,} 个字段被跳过，具体原因见原始工具数据。")
        notes.append("两种指标各自在对应的字段类型内比较，结果描述当前样本里的关联。")
    if pair := evidence.get("pairwise_correlation"):
        r = "无法定义" if pair["r"] is None else f"{pair['r']:+.3f}"
        notes.append(f"使用 {pair['paired_rows']:,} 对有效值，当前表共 {pair['working_rows']:,} 行。Pearson r 为 {r}。")
    if groups := evidence.get("group_summary"):
        notes.append(f"使用 {groups['rows_used']:,} / {groups['working_rows']:,} 行，共得到 {groups['group_count']:,} 组。")
        tables.append({"title": "分组统计结果", "rows": [
            {"分组": row["group"], "数值": row["value"]} for row in groups["groups"]
        ]})
    if trend := evidence.get("trend_summary"):
        notes.append(
            f"用 {trend['rows_used']:,} 行汇总了 {trend['periods']:,} 个时间段。"
            f"范围是 {trend['start_period']} 至 {trend['end_period']}。"
        )
    if distribution := evidence.get("distribution_summary"):
        notes.append(
            f"使用 {distribution['rows_used']:,} / {distribution['working_rows']:,} 行。"
            f"中位数为 {distribution['median']:,.8g}，范围为 {distribution['min']:,.8g} 至 {distribution['max']:,.8g}。"
        )
    if importance := evidence.get("feature_importance"):
        folds = importance["folds"]
        notes.append(
            f"使用 {importance['rows']:,} 行进行 {folds} 折验证。模型误差为 {importance['validation_mae']:,.4g}，"
            f"中位数基准为 {importance['baseline_mae']:,.4g}，模型在 {importance['baseline_wins']}/{folds} 折优于基准。"
        )
        tables.append({"title": "图中单列因素的验证结果", "rows": [
            {"因素": row["field"], "占比": f"{row['share']:.1%}", "打乱后误差上升的折数": f"{row['positive_folds']}/{folds}"}
            for row in importance["factors"]
        ]})
    if evidence.get("png_created"):
        notes.append("已生成 PNG 图表。")
    return notes, tables


def describe_journey(steps: list[dict]) -> list[dict]:
    journey = []
    for step in steps:
        name = step["tool"]
        action = describe_call(name, step["parameters"])
        notes, tables = summarize_evidence(step.get("evidence", {}))
        status = "已完成"
        if step.get("deferred"):
            status = "已延后"
            notes = ["先完成同一轮的数据预处理，再调用这一步。"]
        elif step["status"] == "error":
            status = "工具出错"
            notes = ["这一步遇到错误，具体原因可在原始工具数据里查看。"]
        elif not notes:
            notes = ["工具返回了结果，全文可在原始工具数据里查看。"]
        if review := step.get("evidence", {}).get("candidate_review"):
            counts = review_counts(review)
            notes.insert(0, f"候选字段审查 · {counts['included']} 个入选，{counts['excluded']} 个跳过，{counts['needs_confirmation']} 个待确认。")
        journey.append({
            "round": step.get("round"), "action": action.label if action else "调用分析工具", "status": status,
            "purpose": PURPOSES.get(name, ""), "method": list(action.details) if action else [],
            "outcome": notes, "tables": tables,
        })
    return journey
