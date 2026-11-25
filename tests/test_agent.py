"""Exercise the LangGraph tool loop and PNG reports without API credentials."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langchain.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.agent import MAX_TOOL_TURNS, build_agent, build_model, print_answer, tool_artifacts, tool_trace
from src.tools import charts


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def bind_tools(self, tools, **kwargs):
        for tool in tools:
            assert "runtime" not in tool.tool_call_schema.model_json_schema()["properties"]
        return self

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        self.last_messages = messages
        return next(self.responses)


def call_tool(name, call_id, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_ollama_can_use_an_installed_model_without_an_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
    model = build_model()
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen3.5:4b"
    assert model.base_url == "http://127.0.0.1:11435"
    assert build_model("ollama", "another-model:latest").model == "another-model:latest"


def test_openai_override_requires_a_key_and_uses_the_requested_model(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_model("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    model = build_model("openai", "other-api-model")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "other-api-model"


def test_multiple_pngs(tmp_path, capsys):
    csv = tmp_path / "sales.csv"
    csv.write_text(
        "date,office,revenue,units\n"
        "2025-01-02,East,10,1\n2025-01-17,East,20,2\n"
        "2025-03,West,40,4\ninvalid,West,50,5\n2025-03-09,West,inf,invalid\n",
        encoding="utf-8",
    )
    output = tmp_path / "images"
    model = ScriptedModel([
        call_tool("group_report", "1", group_column="office", value_column="revenue"),
        call_tool("trend_report", "2", date_column="date", value_column="revenue", aggregation="sum"),
        call_tool("distribution_report", "3", value_column="revenue", bins=4),
        call_tool("relationship_report", "4", x_column="units", y_column="revenue"),
        AIMessage(content="已生成四份报告。"),
    ])
    agent = build_agent(model, csv_path=csv, output_dir=output)
    question = "比较各办公室平均营收，给出月度总营收、营收分布及销量与营收关系图。"
    print_answer(agent, question)

    console = capsys.readouterr().out
    assert "已生成四份报告。" in console
    assert "Rows: 5" in model.prompts[0] and "revenue" in model.prompts[0]
    results = [message for message in model.last_messages if isinstance(message, ToolMessage)]
    assert [message.content for message in results if message.status == "error"] == []
    assert "First 2025-01 = 30" in results[1].content and "Last 2025-03 = 40" in results[1].content
    assert "Median 30" in results[2].content
    assert "Pearson r = 1.000" in results[3].content
    files = list(output.glob("*.png"))
    assert {path.stem.split("-")[0] for path in files} == {"group", "trend", "distribution", "relationship"}
    assert len(files) == 4
    for path in files:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert str(path) in console


def test_failed_tool_call_can_be_corrected(tmp_path):
    model = ScriptedModel([
        call_tool("group_report", "1", group_column="invalid", value_column="resale_price"),
        call_tool("group_report", "2", group_column="town", value_column="resale_price"),
        AIMessage(content="BISHAN has the highest average price."),
    ])
    agent = build_agent(model, output_dir=tmp_path)
    result = agent.invoke({"messages": [{"role": "user", "content": "Chart average resale price by town."}]})
    error, success = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert error.status == "error" and error.artifact is None
    assert "Invalid columns" in error.content
    assert success.status == "success" and "820000" in success.content
    assert Path(success.artifact["path"]).is_file()
    assert len(list(tmp_path.iterdir())) == 1
    failed_step, corrected_step = tool_trace(result["messages"])
    assert failed_step["status"] == "error" and failed_step["parameters"]["group_column"] == "invalid"
    assert corrected_step["parameters"]["aggregation"] == "mean"
    assert corrected_step["parameters"]["top_n"] == 10
    assert "820000" in corrected_step["result"]


def test_rank_relationships_compares_lease_area_and_town(tmp_path):
    model = ScriptedModel([
        call_tool("rank_relationships", "compare", target_column="resale_price"),
        AIMessage(content="Among the numeric fields, remaining lease ranks highest."),
    ])
    agent = build_agent(model, output_dir=tmp_path)
    result = agent.invoke({"messages": [{"role": "user", "content": "那房子二手价格和什么关系最大呢"}]})
    step, = tool_trace(result["messages"])
    assert step["tool"] == "rank_relationships" and step["status"] == "success"
    assert step["parameters"]["candidate_columns"] is None
    report = step["result"]
    assert report.index("lease_years_remaining: Pearson r = +0.481, n = 6") < report.index(
        "floor_area_sqm: Pearson r = +0.110, n = 6"
    )
    assert "town: eta = 0.954, n = 6, groups = 3" in report


def test_rank_relationships_orders_absolute_r_and_skips_unusable_fields(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text(
        "target,negative,weak,constant,sparse,label,identifier\n"
        "1,6,1,1,1,A,id1\n2,5,3,1,inf,A,id2\n3,4,2,1,3,A,id3\n"
        "4,3,2,1,,B,id4\n5,2,3,1,,B,id5\n6,1,1,1,,B,id6\n",
        encoding="utf-8",
    )
    model = ScriptedModel([
        call_tool("rank_relationships", "compare", target_column="target"),
        AIMessage(content="Analysis complete."),
    ])
    agent = build_agent(model, csv_path=csv, output_dir=tmp_path)
    result = agent.invoke({"messages": [{"role": "user", "content": "Which field is most related to target?"}]})
    report = tool_trace(result["messages"])[0]["result"]
    assert report.index("negative: Pearson r = -1.000") < report.index("weak: Pearson r")
    assert "label: eta = " in report
    assert "constant: Pearson r requires" in report
    assert "sparse: Pearson r requires" in report
    assert "identifier: eta requires" in report


def test_top_factor_chart(tmp_path):
    random = np.random.default_rng(71)
    count = 240
    volume = random.uniform(5, 30, count)
    item_type = random.choice(["basic", "premium"], count)
    delivery = random.uniform(1, 5, count)
    price = 100 + 18 * volume + np.where(item_type == "premium", 130, 0) + 16 * delivery
    csv = tmp_path / "products.csv"
    pd.DataFrame({
        "volume": volume, "item_type": item_type, "delivery": delivery,
        "constant": 1, "price": price,
    }).to_csv(csv, index=False)
    model = ScriptedModel([
        call_tool("feature_importance_report", "importance", target_column="price", top_n=1),
        AIMessage(content="The report is ready."),
    ])
    agent = build_agent(model, csv_path=csv, output_dir=tmp_path)
    result = agent.invoke({"messages": [{"role": "user", "content": "Plot the most important factor and its share."}]})
    step, = tool_trace(result["messages"])
    assert step["status"] == "success"
    assert step["parameters"]["top_n"] == 1
    assert "3-fold validation | 240 rows" in step["result"]
    assert "- volume:" in step["result"] and "Other factors (2)" in step["result"]
    assert "Skipped: constant" in step["result"]
    tool_message = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    evidence = tool_message.artifact["feature_importance"]
    assert evidence["baseline_wins"] == 3
    assert evidence["factors"][0]["positive_folds"] == 3
    assert evidence["factors"][0]["leading_folds"] == 3
    percentages = re.findall(r"(\d+\.\d)%", step["result"])
    assert len(percentages) == 2 and sum(map(float, percentages)) == pytest.approx(100.0, abs=0.1)
    png, = tmp_path.glob("importance-*.png")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_feature_importance_requires_validation_rows(tmp_path):
    model = ScriptedModel([
        call_tool("feature_importance_report", "importance", target_column="resale_price"),
        AIMessage(content="The sample is too small for predictive importance."),
    ])
    agent = build_agent(model, output_dir=tmp_path)
    result = agent.invoke({"messages": [{"role": "user", "content": "Show feature importance for the demo."}]})
    step, = tool_trace(result["messages"])
    assert step["status"] == "error" and "working data have 6" in step["result"]
    assert not list(tmp_path.glob("*.png"))


def test_filter_and_report(tmp_path):
    csv = tmp_path / "products.csv"
    csv.write_text(
        "market,category,unit_price\nSingapore,hardware,10\n singapore ,hardware,30\nMalaysia,hardware,1000\n",
        encoding="utf-8",
    )
    chart = call_tool("group_report", "early", group_column="category", value_column="unit_price")
    selection = call_tool("filter_rows", "filter", column_name="market", operator="equals", value="Singapore")
    model = ScriptedModel([
        AIMessage(content="", tool_calls=chart.tool_calls + selection.tool_calls),
        call_tool("group_report", "ready", group_column="category", value_column="unit_price"),
        AIMessage(content="Singapore hardware averages 20."),
    ])
    agent = build_agent(model, csv_path=csv, output_dir=tmp_path)
    result = agent.invoke({"messages": [{"role": "user", "content": "只看 market = Singapore，再按 category 比较 unit_price 的均值。"}]})
    report = tool_trace(result["messages"])
    deferred = next(step for step in report if "Deferred" in step["result"])
    assert deferred["status"] == "error" and deferred["deferred"]
    filter_step = next(step for step in report if step["tool"] == "filter_rows")
    assert filter_step["round"] == deferred["round"] == 1
    assert filter_step["evidence"]["row_change"] == {"before": 3, "after": 2}
    completed = [step for step in report if step["tool"] == "group_report" and step["status"] == "success"]
    assert len(completed) == 1 and "20.0" in completed[0]["result"]
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert result["prepared"].frame["unit_price"].tolist() == [10, 30]
    assert "1000" in csv.read_text(encoding="utf-8")
    attempt, = result["validation"]
    assert attempt["status"] == "passed" and not attempt["retrying"]
    assert any("有 2 行满足" in check["detail"] for check in attempt["checks"])
    assert any("hardware 20" in check["detail"] for check in attempt["checks"])


@pytest.mark.parametrize("recover", [True, False])
def test_validation_rejects_chart_for_wrong_scope_and_retries_once(tmp_path, recover):
    csv = tmp_path / "products.csv"
    csv.write_text("market,category,unit_price\nSingapore,hardware,10\nSingapore,hardware,30\nMalaysia,hardware,1000\n", encoding="utf-8")
    selection = call_tool("filter_rows", "select", column_name="market", operator="equals", value="Singapore")
    first = call_tool("group_report", "wrong", group_column="category", value_column="unit_price")
    revised = call_tool("group_report", "corrected", group_column="category", value_column="unit_price")
    model = ScriptedModel([
        first, AIMessage(content="均价是 346.67。"),
        *([selection, revised] if recover else []), AIMessage(content="均价是 20。"),
    ])
    graph = build_agent(model, csv_path=csv, output_dir=tmp_path)
    events = list(graph.stream(
        {"messages": [{"role": "user", "content": "只看 market = Singapore，按 category 画 unit_price 的均价图。"}]},
        stream_mode=["values", "custom"],
    ))
    stages = [event["stage"] for mode, event in events if mode == "custom"]
    assert stages == ["scope", "numbers", "conclusion", "retry", "scope", "numbers", "conclusion"]
    result = next(event for mode, event in reversed(events) if mode == "values")
    initial, final = result["validation"]
    assert initial["status"] == "failed" and initial["retrying"]
    assert any("3 行中，有 2 行" in check["detail"] for check in initial["checks"])
    assert final["status"] == ("passed" if recover else "failed") and not final["retrying"]
    assert result["rejected_tool_calls"] == ["wrong"]
    assert all(message.text != "均价是 346.67。" for message in result["messages"])
    assert "Validation excluded this result" in next(msg.text for msg in model.last_messages if isinstance(msg, ToolMessage))
    assert "Validation rejected the previous draft" in model.prompts[-1]
    artifacts = list(tool_artifacts(result["messages"], result["rejected_tool_calls"]))
    if recover:
        report, = (artifact for artifact in artifacts if "path" in artifact)
        assert report["group_summary"]["groups"] == [{"group": "hardware", "value": 20.0}]
        assert result["messages"][-1].text == "均价是 20。"
    else:
        assert all("path" not in artifact for artifact in artifacts)
        assert "校验未通过" in result["messages"][-1].text


def test_validation_retries_causal_wording_without_recomputing_tool_calls(tmp_path):
    model = ScriptedModel([
        call_tool("relationship_report", "pair", x_column="floor_area_sqm", y_column="resale_price"),
        AIMessage(content="面积显著影响价格。但是这不能证明因果。"),
        AIMessage(content="面积与价格的 Pearson r 约为 0.110。这不能用来证明面积导致价格变化。"),
    ])
    result = build_agent(model, output_dir=tmp_path).invoke({"messages": [{"role": "user", "content": "画图比较面积与价格。"}]})
    assert [attempt["status"] for attempt in result["validation"]] == ["failed", "passed"]
    assert result["validation"][0]["checks"][-1]["label"] == "结论措辞"
    assert result["rejected_tool_calls"] == []
    assert any("Pearson r，结果为 +0.110" in check["detail"] for check in result["validation"][-1]["checks"])


def test_validation_detects_a_wrong_tool_number(tmp_path, monkeypatch):
    summarize = charts.summarize_groups
    wrong_first = iter([True, False])

    def summarize_with_error(groups, values, aggregation):
        result = summarize(groups, values, aggregation)
        return result + 100 if next(wrong_first) else result

    monkeypatch.setattr(charts, "summarize_groups", summarize_with_error)
    model = ScriptedModel([
        call_tool("average_by_group", "wrong", group_column="town", value_column="resale_price"),
        AIMessage(content="均值已计算。"),
        call_tool("average_by_group", "corrected", group_column="town", value_column="resale_price"),
        AIMessage(content="已重新计算均值。"),
    ])
    result = build_agent(model, output_dir=tmp_path).invoke({"messages": [{"role": "user", "content": "按 town 比较平均转售价。"}]})
    assert [attempt["status"] for attempt in result["validation"]] == ["failed", "passed"]
    errors = [check for check in result["validation"][0]["checks"] if check["status"] == "failed"]
    error, = errors
    assert error["label"] == "独立复算" and "BISHAN 820,000" in error["detail"]
    assert result["rejected_tool_calls"] == ["wrong"]


def test_clean_and_restore(tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "day,team,revenue\n01/02/2025,A,10\n01/02/2025,A,10\n02/02/2025,B,bad\n03/02/2025, ,40\n",
        encoding="utf-8",
    )
    model = ScriptedModel([
        call_tool("profile_dataset", "profile"),
        call_tool("convert_column", "date", column_name="day", data_type="date", date_format="%d/%m/%Y"),
        call_tool("convert_column", "number", column_name="revenue", data_type="number"),
        call_tool("drop_missing", "missing", columns=["team", "revenue"]),
        call_tool("drop_duplicates", "dup"),
        AIMessage(content="One clean row remains."),
        call_tool("reset_dataset", "reset"),
        AIMessage(content="All rows restored."),
    ])
    agent = build_agent(model, csv_path=csv, output_dir=tmp_path)
    first = agent.invoke({"messages": [{"role": "user", "content": "Prepare the data."}]})
    working = first["prepared"]
    assert working.frame["revenue"].tolist() == [10.0]
    assert working.frame["day"].iloc[0].month == 2
    assert "1 nonblank values could not be parsed" in working.steps[1]["detail"]
    assert [step["rows_after"] for step in working.steps] == [4, 4, 2, 1]
    restored = agent.invoke({"messages": [{"role": "user", "content": "Restore all rows."}], "prepared": working})
    assert restored["prepared"] is None
    assert "4 rows" in tool_trace(restored["messages"])[0]["result"]


def test_failed_or_empty_filter_never_falls_back_to_all_rows(tmp_path):
    model = ScriptedModel([
        call_tool("filter_rows", "bad", column_name="missing_country", operator="equals", value="Singapore"),
        call_tool("group_report", "blocked", group_column="town", value_column="resale_price"),
        call_tool("filter_rows", "empty", column_name="town", operator="equals", value="no matches"),
        call_tool("group_report", "no-data", group_column="town", value_column="resale_price"),
        AIMessage(content="There are no matching rows."),
    ])
    agent = build_agent(model, output_dir=tmp_path)
    result = agent.invoke({"messages": [{"role": "user", "content": "Filter the data and then draw a chart."}]})
    failed, blocked, empty, chart = tool_trace(result["messages"])
    assert failed["status"] == blocked["status"] == chart["status"] == "error"
    assert "Resolve the failed step" in blocked["result"]
    assert empty["status"] == "success" and result["prepared"].frame.empty
    assert not list(tmp_path.iterdir())


def test_tool_loop_finishes_at_the_budget(tmp_path):
    model = ScriptedModel([
        *(call_tool("dataset_overview", str(turn)) for turn in range(MAX_TOOL_TURNS)),
        AIMessage(content="The CSV contains six rows."),
    ])
    agent = build_agent(model, output_dir=tmp_path)
    result = agent.invoke({"messages": [{"role": "user", "content": "Show the row count."}]})
    assert result["tool_turns"] == MAX_TOOL_TURNS
    assert "budget is exhausted" in model.prompts[-1]
    assert result["messages"][-1].content == "The CSV contains six rows."
    assert not list(tmp_path.iterdir())
