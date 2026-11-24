# Agentic Data Analyst

A data analyst built with **LangChain**, **LangGraph**, **Pandas**, and **Matplotlib**, with a local browser interface and a CLI. Ask a question in natural language. The agent chooses the analysis functions, summarizes their results, and generates PNG reports.

## Current functionality (MVP)

- Inspect the schema, row count, and actual rows of the demo data or your own CSV.
- Choose an OpenAI API model or a local Ollama model with tool-calling support.
- Compare groups by mean, median, sum, or row count with a bar chart.
- Plot a metric over time, its distribution, or its relationship to another numeric metric.
- Generate multiple reports from one request, or ask for a text-only answer.
- Inspect completed tool calls and download their parameters, outcomes, and dataset hash as JSON.

> The included HDB-style CSV contains **fictional synthetic rows for testing**, not real Singapore resale transactions. Replace it with a properly licensed public dataset when ready.

## Quick start

From the repository root:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Browser interface

Start Ollama for local inference, then launch the interface from the project directory:

```bash
python -m streamlit run app.py
```

Open `http://127.0.0.1:8501`. The sidebar lets you upload a CSV, inspect its fields, and choose Ollama or OpenAI. The demo data and local `qwen3.5:4b` are selected initially unless you have set model defaults. An OpenAI key can come from the password field or `OPENAI_API_KEY` in your environment. Try an example question or type one below the conversation. Each report can be downloaded as a PNG from its message.

Open **预览数据** above the chat to inspect the first ten rows without asking a model. You can also ask to see the current dataset in the chat. **查看执行记录** below a tool-backed reply shows the tools that returned results or errors and offers a JSON download.

While an answer is running, an animated analyst types at a workstation and the current action appears in a thought bubble. Each bubble describes the selected analysis, including its fields and parameters. Meaningful steps move into the conversation history, while the initial waiting cue disappears. Browser reduced-motion preferences disable the animations.

The latest four completed exchanges are available for follow-up questions. A new conversation starts when you change the dataset or model provider. Browser conversations and generated reports are kept in the session. Download any images you want to keep.

### Command-line interface

#### Local Ollama

Start Ollama and check that the model you want appears in `ollama list`. The agent defaults to `qwen3.5:4b` in Ollama mode. To choose any installed model that supports tool calling, pass its exact tag to `--model`.

```bash
python -m src.agent --provider ollama --model qwen3.5:4b --question "Compare average resale price by town and plot the distribution of resale prices."
```

This mode needs no OpenAI API key. It connects to `http://127.0.0.1:11434` by default. Set `OLLAMA_BASE_URL` in `.env` if your Ollama server runs at a different address. Ollama manages model storage through its own settings, including `OLLAMA_MODELS`.

#### OpenAI API

Set `OPENAI_API_KEY` in your environment or copy `.env.example` to `.env` and set it there. OpenAI remains the default provider and defaults to `gpt-4o-mini`.

```bash
python -m src.agent --provider openai --model gpt-4o-mini --question "Compare average resale price by town."
```

#### Defaults and data files

Set `MODEL_PROVIDER` to `ollama` or `openai` in `.env` to choose the default for your workspace. Set `OLLAMA_MODEL` or `OPENAI_MODEL` to save a default model for that provider. `--provider` and `--model` take precedence. Omit `--question` for interactive use. Both providers save PNGs under `reports/` by default.

To analyze a different CSV and choose where to save the images:

```bash
python -m src.agent --csv path/to/sales.csv --output-dir reports/sales --question "Plot monthly total revenue."
```

The demo CSV supports group comparisons, histograms, and scatter plots. Time trends need a CSV with a date field such as `month` or `transaction_date`. Requests for filters and custom formulas are outside the current tool set. Each interactive question runs independently, so include any context you want the analysis to use.

## How it works

LangChain binds the analysis tools to the chat model. LangGraph routes simple requests to browse the dataset straight to the preview tool. For analysis requests, the `analyst` node supplies the schema and routes the model's tool calls to `ToolNode`. Results return to the model so it can choose another tool or summarize the analysis. A run allows up to six rounds of tool calls. Pandas calculates the statistics and Matplotlib renders the images. The CLI reads the actual file paths from the tool results.

Run tests without making LLM API calls:

```bash
python -m pytest -q
```

## Roadmap

1. Add an independent validation step for grouped statistics and use the outcome to route failures to a bounded recovery step.
2. Replay recorded tool calls against a CSV with the matching SHA-256 hash, without calling a model.
3. Benchmark tool selection and result accuracy on known answers, including missing values and invalid columns.

## Safety and reproducibility

- Treat CSVs as read-only inputs. The CLI saves images to the configured report directory. The browser interface prepares downloads using temporary files.
- The model receives the question, schema, and tool results, which can include category labels and statistics. Choose data you are authorized to share with the model provider.
- Keep credentials out of source control.
- Add dataset source/licensing information when switching from synthetic data to public data.

## References

- [LangChain Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart)
- [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [LangChain ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama)
