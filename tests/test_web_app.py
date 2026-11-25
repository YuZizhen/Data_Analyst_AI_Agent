"""Exercise chat state and report display with the existing analysis tools."""

from pathlib import Path

import pytest
from langchain.messages import AIMessage
from streamlit.testing.v1 import AppTest

from src import agent, table_store
from src.web_progress import Activity, activity_for_update, activity_html
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
    numeric, categorical = [item.value for item in app.dataframe[1:]]
    assert numeric["Pearson r"].tolist() == ["+0.481", "+0.110"]
    assert categorical["相关比 η"].tolist() == ["0.954"]
    assert any("lease_years_remaining: Pearson r = +0.481" in item.value for item in app.code)
    assert any("部分校验" in item.value for item in app.warning)


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
