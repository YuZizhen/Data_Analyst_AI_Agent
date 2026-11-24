"""Exercise the LangGraph tool loop and PNG reports without API credentials."""

from pathlib import Path

import pytest
from langchain.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.agent import MAX_TOOL_TURNS, build_agent, build_model, print_answer, tool_trace


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
