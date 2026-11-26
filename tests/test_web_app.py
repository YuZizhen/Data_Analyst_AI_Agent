"""Exercise chat state and report display with the existing analysis tools."""

from pathlib import Path

import pytest
from langchain.messages import AIMessage
from streamlit.testing.v1 import AppTest

from src import agent, table_store
from src.web_progress import Activity, activity_for_update, activity_html, update_prediction_activity
from test_agent import ScriptedModel, call_tool


@pytest.fixture(autouse=True)
def isolated_table_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(table_store, "DATASET_DIRECTORY", tmp_path / "t")


def test_activity_moves_previous_bubbles_into_the_chat_history():
    steps = [Activity("正在检查字段"), Activity("正在绘图", "chart")]
    running = activity_html(steps, running=True)
    archived, scene = running.split('<div class="activity-scene">')
    assert "正在检查字段" in archived and "正在绘图" not in archived
    assert 'class="analyst-workstation"' in scene and 'role="status"' in scene and "正在绘图" in scene
    finished = activity_html(steps, running=False)
    assert "正在检查字段" in finished and "正在绘图" in finished
    assert 'class="analyst-workstation"' not in finished and 'role="status"' not in finished


def test_progress_combines_simultaneous_tool_calls_and_ignores_result_events():
    group = call_tool("group_report", "1", group_column="town", value_column="resale_price")
    histogram = call_tool("distribution_report", "2", value_column="resale_price", bins=5)
    batch = AIMessage(content="", tool_calls=group.tool_calls + histogram.tool_calls)
    bubble, = activity_for_update({"analyst": {"messages": [batch]}})
    assert bubble.label == "同时执行 2 项分析"
    assert "按「town」分组，计算「resale_price」的平均值" in bubble.details[0]
    assert "5 个等宽区间" in bubble.details[1]
    assert not list(activity_for_update({"tools": {"messages": []}}))
    filter_call = call_tool("filter_rows", "3", column_name="market", operator="equals", value="Singapore")
    mixed = AIMessage(content="", tool_calls=group.tool_calls + filter_call.tool_calls)
    step, = activity_for_update({"analyst": {"messages": [mixed]}})
    assert step.label == "筛选数据" and "Singapore" in step.details[0]


def test_prediction_progress_updates_each_stage_in_one_bubble():
    activity = []
    event = {"event": "prediction_progress", "target": "price", "tool_call_id": "model", "folds": 3,
             "rows": 12000, "available_rows": 162691}
    for stage in ("fit", "importance"):
        for fold in (1, 2, 3):
            assert update_prediction_activity(activity, {**event, "stage": stage, "fold": fold})
    assert update_prediction_activity(activity, {**event, "stage": "chart"})
    trained, scored, drawing = activity
    assert "第 3/3 折" in trained.details[0] and "162,691" in trained.details[0]
    assert "第 3/3 折" in scored.details[0]
    assert drawing.label == "正在绘制因素重要性图" and drawing.phase == "chart"


def test_browser_chat_keeps_context_and_resets_on_provider_change(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    model = ScriptedModel([
        call_tool("group_report", "plot", group_column="town", value_column="resale_price"),
        AIMessage(content="BISHAN 的平均转售价最高，为 820,000。"),
        AIMessage(content="820,000。"),
    ])
    monkeypatch.setattr(agent, "build_model", lambda *args, **kwargs: model)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=25).run()

    app.chat_input[0].set_value("画一张各区平均转售价图。").run()
    app.run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert len(app.get("download_button")) == 2
    assert app.session_state.turns[0].reports[0].image.startswith(b"\x89PNG\r\n\x1a\n")
    step, validation = app.session_state.turns[0].activity
    assert step.label == "绘制分组柱状图"
    assert "按「town」分组，计算「resale_price」的平均值" in step.details[0]
    assert validation.label == "正在校验" and any("独立复算" in item for item in validation.details)
    assert any(item.label == "校验结果" for item in app.tabs)
    assert any(item.value == "校验通过" for item in app.success)
    assert any("用 Python 逐组重算了" in item.value for item in app.caption)
    assert any("请求的筛选" in item.value for item in app.text)
    assert any("实际生效的筛选" in item.value for item in app.text)
    assert app.session_state.turns[0].trace["validation"][0]["status"] == "passed"

    app.chat_input[0].set_value("上一轮图里最高的数值是多少？").run()
    app.run()
    assert not app.exception
    assert len(app.chat_message) == 4
    assert any("BISHAN" in message.text for message in model.last_messages)
    assert len(app.get("download_button")) == 2
    assert len(app.session_state.turns[0].activity) == 2
    assert not app.session_state.turns[1].activity

    app.radio[0].set_value("openai").run()
    assert not app.exception
    assert not app.chat_message
    assert any(widget.label == "OpenAI API Key" for widget in app.text_input)


def test_browser_previews_actual_rows_and_records_tool_response(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    model = ScriptedModel([])
    monkeypatch.setattr(agent, "build_model", lambda *args, **kwargs: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=25).run()
    assert any(expander.label == "预览数据" for expander in app.expander)
    assert len(app.dataframe[0].value) == 6

    app.chat_input[0].set_value("我可以看一下这个数据库吗？").run()
    app.run()
    assert not app.exception
    assert app.dataframe[1].value["town"].tolist() == app.dataframe[0].value["town"].tolist()
    assert not model.prompts
    step, = app.session_state.turns[0].activity
    assert step.label == "预览数据" and "最多 10 行" in step.details[0]
    trace = app.session_state.turns[0].trace
    assert len(trace["dataset"]["sha256"]) == 64
    assert "resale_price" in trace["dataset"]["columns"]
    assert trace["steps"] == [{
        "round": 1, "tool": "dataset_preview", "parameters": {"limit": 10}, "status": "success",
        "result": "Dataset preview ready for display. Showing 6 of 6 rows.",
        "evidence": {"preview": {"rows": 6, "total_rows": 6}},
    }]


def test_preview_shows_the_uploaded_rows_even_if_model_invents_a_table(monkeypatch):
    csv = b"date,product,units_sold\n2025-01-05,A101,45\n2025-01-06,B202,12\n"
    entry = table_store.TableStore(table_store.DATASET_DIRECTORY).save_original("sales.csv", csv)
    invented = "Here is the preview:\n\n| date | product | units_sold |\n| --- | --- | --- |\n| 2024-01-01 | Laptop | 5 |"
    model = ScriptedModel([call_tool("dataset_preview", "preview"), AIMessage(content=invented)])
    monkeypatch.setattr(agent, "build_model", lambda *args, **kwargs: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=25).run()
    app.selectbox[0].set_value(entry.key).run()
    app.chat_input[0].set_value("Let me take a look at the sales records.").run()
    app.run()
    assert not app.exception
    assert app.dataframe[0].value.equals(app.dataframe[1].value)
    assert app.dataframe[1].value["product"].tolist() == ["A101", "B202"]
    assert all("Laptop" not in item.value for item in app.markdown)

    app.chat_input[0].set_value("preview").run()
    app.run()
    assert not app.exception and len(model.prompts) == 2
    assert app.dataframe[0].value.equals(app.dataframe[2].value)


def test_combined_preview_keeps_analysis_without_duplicate_markdown_table(tmp_path):
    csv = tmp_path / "sales.csv"
    csv.write_text("product,units_sold\nA101,45\nB202,12\n", encoding="utf-8")
    model = ScriptedModel([
        call_tool("dataset_preview", "preview"), call_tool("count_rows", "count"),
        AIMessage(content="| product | units_sold |\n| --- | --- |\n| Laptop | 5 |\n\nThere are 2 rows."),
    ])
    result = agent.build_agent(model, csv_path=csv, output_dir=tmp_path).invoke({
        "messages": [{"role": "user", "content": "Show me the sales records and count them."}],
    })
    assert result["messages"][-1].text == "There are 2 rows."
    preview, count = agent.tool_artifacts(result["messages"])
    assert preview["table"]["data"] == [["A101", 45], ["B202", 12]]
    assert count["row_count"]["rows"] == 2


def test_browser_reports_an_empty_reply_without_claiming_analysis_finished(monkeypatch):
    model = ScriptedModel([AIMessage(content=""), AIMessage(content="\n")])
    monkeypatch.setattr(agent, "build_model", lambda *args, **kwargs: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=25).run()
    app.chat_input[0].set_value("和什么相关呢").run()
    app.run()
    assert not app.exception
    assert any("本轮没有执行分析" in item.value for item in app.markdown)
    assert all("分析已完成" not in item.value for item in app.markdown)
    assert len(model.prompts) == 2 and app.session_state.turns[0].trace is None


def test_sales_conversation_answers_the_total_and_handles_a_correction(monkeypatch):
    csv = ("region,units_sold\nNorth,45\nSouth,12\nEast,30\nWest,88\nNorth,20\nSouth,15\n"
           "East,50\nWest,8\nNorth,60\nSouth,40\n")
    entry = table_store.TableStore(table_store.DATASET_DIRECTORY).save_original("sales.csv", csv.encode())
    model = ScriptedModel([
        call_tool("dataset_overview", "schema"), AIMessage(content="10 rows and 2 columns."),
    ])
    monkeypatch.setattr(agent, "build_model", lambda *args, **kwargs: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=25).run()
    app.selectbox[0].set_value(entry.key).run()
    for question in ("Show me a preview of the data", "How many rows are there and what columns do we have?",
                     "What is the total units sold across all records?", "it is not a summary"):
        app.chat_input[0].set_value(question).run()
        app.run()
        assert not app.exception
    turns = app.session_state.turns
    assert [turn.trace["steps"][-1]["tool"] for turn in turns] == [
        "dataset_preview", "dataset_overview", "aggregate_column", "aggregate_column",
    ]
    assert all("**368**" in turn.messages[-1].text and turn.trace["validation"][-1]["status"] == "passed"
               for turn in turns[-2:])


def test_browser_renders_the_filtered_rows_and_the_formula_totals(monkeypatch):
    csv = (
        "category,units_sold,unit_price,discount\nElectronics,45,120,.05\nFurniture,12,85,0\n"
        "Electronics,30,120,.1\nClothing,88,45,.15\nFurniture,20,85,0\nElectronics,15,115,.05\n"
        "Clothing,50,45,.2\nFurniture,8,90,0\nElectronics,60,110,.08\nClothing,40,42,.1\n"
    )
    entry = table_store.TableStore(table_store.DATASET_DIRECTORY).save_original("sales.csv", csv.encode())
    model = ScriptedModel([])
    monkeypatch.setattr(agent, "build_model", lambda *args, **kwargs: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=25).run()
    app.selectbox[0].set_value(entry.key).run()
    app.chat_input[0].set_value("Only show rows where category is Electronics. What's the average unit price?").run()
    app.run()
    assert not app.exception
    turn, = app.session_state.turns
    assert [step["tool"] for step in turn.trace["steps"]] == ["filter_rows", "dataset_preview", "aggregate_column"]
    assert any(item.value["category"].tolist() == ["Electronics"] * 4 for item in app.dataframe if "category" in item.value)
    assert "**116.25**" in turn.messages[-1].text
    assert turn.trace["validation"][-1]["status"] == "passed"

    app.selectbox[0].set_value(entry.key).run()
    app.chat_input[0].set_value("Which category has the highest total revenue? (revenue = units_sold × unit_price × (1 − discount))").run()
    app.run()
    assert not app.exception
    turn = app.session_state.turns[-1]
    step, = turn.trace["steps"]
    assert step["tool"] == "formula_group_totals"
    assert "**16,080.75** (Electronics)" in turn.messages[-1].text
    assert [round(row["value"], 2) for row in step["evidence"]["group_summary"]["groups"]] == [16080.75, 6678, 3440]
    assert turn.trace["validation"][-1]["status"] == "passed"
    assert not model.prompts


def test_journey_explains_the_relationship_tool_with_its_actual_results(monkeypatch):
    model = ScriptedModel([
        call_tool("rank_relationships", "rank", target_column="resale_price"),
        AIMessage(content="下面是字段关联排名。"),
    ])
    monkeypatch.setattr(agent, "build_model", lambda *args, **kwargs: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=25).run()
    app.chat_input[0].set_value("哪个字段与转售价关联最强？").run()
    app.run()
    assert not app.exception
    assert any("在 6 行中检查了 3 个候选字段" in item.value for item in app.text)
    assert any("多个候选字段逐一和同一个目标列比较" in item.value for item in app.text)
    numeric, categorical, review = [item.value for item in app.dataframe[1:]]
    assert numeric["Pearson r"].tolist() == ["+0.481", "+0.110"]
    assert categorical["相关比 η"].tolist() == ["0.954"]
    assert any("lease_years_remaining: Pearson r = +0.481" in item.value for item in app.code)
    assert any("部分校验" in item.value for item in app.warning)
    assert any(tab.label == "候选字段审查" for tab in app.tabs)
    assert set(review["字段"]) == {"lease_years_remaining", "floor_area_sqm", "town"}
    assert review["审查结果"].tolist() == ["入选"] * 3
    review_step, calculate_step, *_ = app.session_state.turns[0].activity
    assert review_step.label == "正在审查候选字段" and "入选 3 个" in review_step.details[-1]
    assert calculate_step.label == "比较字段关联强度"


def test_browser_shows_prediction_progress_and_measured_error(monkeypatch):
    csv = "size,price\n" + "".join(f"{index},{index * 5 + 80}\n" for index in range(60))
    entry = table_store.TableStore(table_store.DATASET_DIRECTORY).save_original("prices.csv", csv.encode())
    model = ScriptedModel([
        call_tool("evaluate_prediction", "prediction", target_column="price"),
        AIMessage(content="预测误差低于中位数基准。"),
    ])
    monkeypatch.setattr(agent, "build_model", lambda *args, **kwargs: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=25).run()
    app.selectbox[0].set_value(entry.key).run()
    app.chat_input[0].set_value("那我可以预测二手售卖的价格吗？").run()
    app.run()
    assert not app.exception
    turn = app.session_state.turns[0]
    step, = turn.trace["steps"]
    assert step["tool"] == "evaluate_prediction" and step["status"] == "success"
    assert any(item.label == "正在训练并验证" and "随机训练/测试切分" in item.details[0] for item in turn.activity)
    assert any(item.value == "训练与测试明细" for item in app.caption)
    assert any("模型 MAE" in item.value and "中位数基准" in item.value for item in app.text)
    assert not turn.reports


def test_browser_keeps_prepared_data_until_restored(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    model = ScriptedModel([
        call_tool("filter_rows", "filter", column_name="town", operator="equals", value="bishan"),
        AIMessage(content="已筛选。"),
    ])
    monkeypatch.setattr(agent, "build_model", lambda *args, **kwargs: model)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=25).run()
    original_key = app.selectbox[0].value
    app.chat_input[0].set_value("只保留 Bishan。").run()
    app.run()
    assert not app.exception
    cleaned_key = app.selectbox[0].value
    assert cleaned_key != original_key
    assert app.dataframe[0].value["town"].tolist() == ["BISHAN", "BISHAN"]
    assert any(item.label == "清洗逻辑与代码" for item in app.expander)
    assert any("filter_frame" in item.value for item in app.code)
    app.chat_input[0].set_value("我可以看一下这份数据吗？").run()
    app.run()
    assert not app.exception
    assert app.dataframe[1].value["town"].tolist() == ["BISHAN", "BISHAN"]
    assert app.session_state.turns[1].trace["working_data"]["before"][0]["tool"] == "filter_rows"
    restore = next(button for button in app.button if button.label == "恢复原始数据")
    restore.click().run()
    assert not app.exception
    assert app.selectbox[0].value == original_key
    assert len(app.dataframe[0].value) == 6

    fresh = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=25).run()
    fresh.selectbox[0].set_value(cleaned_key).run()
    assert not fresh.exception
    assert len(fresh.dataframe[0].value) == 2
