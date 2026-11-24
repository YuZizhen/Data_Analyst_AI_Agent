"""Exercise chat state and report display with the existing analysis tools."""

from pathlib import Path

from langchain.messages import AIMessage
from streamlit.testing.v1 import AppTest

from src import agent
from test_agent import ScriptedModel, call_tool


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
    assert len(app.get("download_button")) == 1
    assert app.session_state.turns[0].reports[0].image.startswith(b"\x89PNG\r\n\x1a\n")

    app.chat_input[0].set_value("上一轮图里最高的数值是多少？").run()
    app.run()
    assert not app.exception
    assert len(app.chat_message) == 4
    assert any("BISHAN" in message.text for message in model.last_messages)
    assert len(app.get("download_button")) == 1

    app.radio[0].set_value("openai").run()
    assert not app.exception
    assert not app.chat_message
    assert any(widget.label == "OpenAI API Key" for widget in app.text_input)
