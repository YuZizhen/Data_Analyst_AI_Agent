"""Local browser interface for the data analyst agent."""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from langchain.messages import AnyMessage, HumanMessage

from src.agent import (
    MODEL_ERRORS,
    build_analysis_graph,
    build_model,
    default_model_name,
    model_error_message,
    tool_artifacts,
    tool_trace,
)
from src.tools import DATA_PATH, DEFAULT_PREVIEW_ROWS, AnalysisContext, load_dataset
from src.web_progress import Activity, activity_for_update, activity_html

DEMO_QUESTIONS = {
    "比较各区转售价": "生成各区平均转售价的柱状图，并指出哪个区最高。",
    "查看转售价分布": "生成转售价分布直方图，并给出中位数和范围。",
    "分析面积与价格": "画一张房屋面积与转售价的散点图，并解释相关性。",
}
CONTEXT_TURNS = 4


@dataclass(frozen=True)
class Dataset:
    name: str
    frame: pd.DataFrame | None
    is_demo: bool
    checksum: str


@dataclass(frozen=True)
class Report:
    title: str
    filename: str
    image: bytes


@dataclass(frozen=True)
class Turn:
    question: str
    messages: list[AnyMessage] = field(default_factory=list)
    reports: list[Report] = field(default_factory=list)
    trace: dict | None = None
    error: str = ""
    activity: list[Activity] = field(default_factory=list)


def selected_dataset(upload):
    fingerprint = (upload.name, sha256(upload.getbuffer()).hexdigest()) if upload is not None else None
    cached = st.session_state.get("dataset")
    if cached is not None and cached[0] == fingerprint and hasattr(cached[1], "checksum"):
        return cached

    is_demo = upload is None
    name = f"{DATA_PATH.name} (synthetic demo)" if is_demo else upload.name
    checksum = ""
    try:
        if is_demo:
            data = DATA_PATH.read_bytes()
            checksum = sha256(data).hexdigest()
            frame = load_dataset(BytesIO(data))
        else:
            checksum = fingerprint[1]
            upload.seek(0)
            frame = load_dataset(upload)
        if frame.empty:
            frame = None
    except (OSError, ValueError):
        frame = None
    st.session_state.dataset = (fingerprint, Dataset(name, frame, is_demo, checksum))
    return st.session_state.dataset


def analyze(
    question: str, dataset: Dataset, provider: str, model_name: str, api_key: str,
    show_activity: Callable[[list[Activity]], None],
) -> Turn:
    activity: list[Activity] = []
    show_activity([Activity("正在查看问题和数据字段")])
    try:
        model_name = model_name.strip() or default_model_name(provider)
        model = build_model(provider, model_name, api_key=api_key.strip())
        recent = [turn for turn in st.session_state.turns if turn.messages][-CONTEXT_TURNS:]
        history = [message for turn in recent for message in turn.messages]
        with TemporaryDirectory(prefix="analyst-") as directory:
            context = AnalysisContext(dataset.frame, dataset.name, Path(directory).resolve())
            graph = build_analysis_graph(model, context)
            inputs = {"messages": [*history, HumanMessage(content=question)]}
            result = inputs
            for mode, event in graph.stream(inputs, stream_mode=["updates", "values"]):
                if mode == "values":
                    result = event
                else:
                    for step in activity_for_update(event):
                        if step not in activity:
                            activity.append(step)
                            show_activity(activity)
            messages = result["messages"][len(history):]
            reports = []
            for artifact in tool_artifacts(messages):
                if "path" in artifact:
                    path = Path(artifact["path"])
                    reports.append(Report(artifact["title"], path.name, path.read_bytes()))
        steps = tool_trace(messages)
        trace = None
        if steps:
            trace = {
                "version": 1,
                "dataset": {
                    "name": dataset.name,
                    "sha256": dataset.checksum,
                    "rows": len(dataset.frame),
                    "columns": {name: str(dtype) for name, dtype in dataset.frame.dtypes.items()},
                },
                "question": question,
                "model": {"provider": provider, "name": model_name},
                "steps": steps,
            }
        return Turn(question, messages, reports, trace, activity=activity)
    except MODEL_ERRORS as error:
        if isinstance(error, ConnectionError):
            return Turn(question, error="连接不到 Ollama。请启动 Ollama 后重新发送问题。", activity=activity)
        return Turn(question, error=model_error_message(error), activity=activity)
    except ValueError as error:
        return Turn(question, error=f"无法启动分析。{error}", activity=activity)
    except OSError:
        return Turn(question, error="无法读取生成的图表。请重新发送问题。", activity=activity)


def show_preview(frame: pd.DataFrame, total_rows: int):
    st.caption(f"显示前 {len(frame):,} 行 · 共 {total_rows:,} 行")
    st.dataframe(frame, hide_index=True, width="stretch")


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
        st.markdown(turn.messages[-1].text or "分析已完成。")
        for artifact in tool_artifacts(turn.messages):
            if "table" in artifact:
                show_preview(pd.DataFrame(**artifact["table"]), artifact["total_rows"])
        for report in turn.reports:
            with st.container(border=True):
                st.image(report.image, caption=report.title, width="stretch")
                st.download_button(
                    "下载 PNG", report.image, report.filename, mime="image/png",
                    key=report.filename, on_click="ignore",
                )
        trace = getattr(turn, "trace", None)
        if trace:
            with st.expander("查看执行记录"):
                for step in trace["steps"]:
                    status = "工具返回错误" if step["status"] == "error" else "工具已返回结果"
                    st.text(f"{step['tool']} · {status}")
                    st.json(step["parameters"], expanded=False)
                    st.text(step["result"])
                st.download_button(
                    "下载执行记录 JSON", json.dumps(trace, ensure_ascii=False, indent=2),
                    "analysis-trace.json", mime="application/json",
                    key=f"trace-{turn.messages[0].id}", on_click="ignore",
                )


def main():
    st.set_page_config(page_title="数据分析工作台", page_icon=":material/analytics:", layout="wide")
    st.html(Path(__file__).with_name("web.css"))

    with st.sidebar:
        st.subheader("数据源", anchor=False)
        upload = st.file_uploader("上传 CSV", type="csv")
        fingerprint, dataset = selected_dataset(upload)
        if dataset.frame is None:
            st.error("无法读取数据。请上传有表头和数据行的 UTF-8 CSV 文件。")
        else:
            st.text("HDB 转售示例（合成数据）" if dataset.is_demo else dataset.name)
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

        scope = (fingerprint, provider)
        if st.session_state.get("scope") != scope:
            st.session_state.scope = scope
            st.session_state.turns = []
        st.divider()
        if st.button("新对话", width="stretch", disabled=not st.session_state.turns):
            st.session_state.turns = []

    st.title("数据分析工作台", anchor=False)
    st.markdown("用自然语言分析 CSV，生成可下载的图表。")
    if dataset.frame is not None:
        with st.expander("预览数据"):
            show_preview(dataset.frame.head(DEFAULT_PREVIEW_ROWS), len(dataset.frame))

    example = None
    if not st.session_state.turns and dataset.frame is not None:
        with st.container(border=True, key="start_here"):
            st.subheader("开始分析", anchor=False)
            st.markdown("选择一个示例，或在下方输入自己的问题。")
            prompts = DEMO_QUESTIONS if dataset.is_demo else {"查看数据行": "我可以看一下这份数据吗？"}
            for area, (label, question) in zip(st.columns(len(prompts)), prompts.items()):
                if area.button(label, width="stretch"):
                    example = question

    for turn in st.session_state.turns:
        show_turn(turn)

    question = st.chat_input("输入你的分析问题…", disabled=dataset.frame is None) or example
    if question and question.strip():
        question = question.strip()
        with st.chat_message("user", avatar=":material/person:"):
            st.text(question)
        with st.chat_message("assistant", avatar=":material/analytics:"):
            progress_slot = st.empty()

            def show_activity(steps):
                progress_slot.html(activity_html(steps, running=True))

            turn = analyze(question, dataset, provider, model_name, api_key, show_activity)
        st.session_state.turns.append(turn)
        st.rerun()


if __name__ == "__main__":
    main()
