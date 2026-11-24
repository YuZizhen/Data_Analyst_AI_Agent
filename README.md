# Agentic Data Analyst

A CLI data analyst built with **LangChain**, **LangGraph**, **Pandas**, and **Matplotlib**. Ask a question in natural language. The agent chooses the analysis functions, summarizes their results, and returns the paths to any PNG reports it creates.

## Current functionality (MVP)

- Inspect the schema and row count of the demo data or your own CSV.
- Choose an OpenAI API model or a local Ollama model with tool-calling support.
- Compare groups by mean, median, sum, or row count with a bar chart.
- Plot a metric over time, its distribution, or its relationship to another numeric metric.
- Generate multiple reports from one request, or ask for a text-only answer.

> The included HDB-style CSV contains **fictional synthetic rows for testing**, not real Singapore resale transactions. Replace it with a properly licensed public dataset when ready.

## Quick start

From the repository root:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Local Ollama

Start Ollama and check that the model you want appears in `ollama list`. The agent defaults to `qwen3.5:4b` in Ollama mode. To choose any installed model that supports tool calling, pass its exact tag to `--model`.

```bash
python -m src.agent --provider ollama --model qwen3.5:4b --question "Compare average resale price by town and plot the distribution of resale prices."
```

This mode needs no OpenAI API key. It connects to `http://127.0.0.1:11434` by default. Set `OLLAMA_BASE_URL` in `.env` if your Ollama server runs at a different address. Ollama manages model storage through its own settings, including `OLLAMA_MODELS`.

### OpenAI API

Set `OPENAI_API_KEY` in your environment or copy `.env.example` to `.env` and set it there. OpenAI remains the default provider and defaults to `gpt-4o-mini`.

```bash
python -m src.agent --provider openai --model gpt-4o-mini --question "Compare average resale price by town."
```

### Defaults and data files

Set `MODEL_PROVIDER` to `ollama` or `openai` in `.env` to choose the default for your workspace. Set `OLLAMA_MODEL` or `OPENAI_MODEL` to save a default model for that provider. `--provider` and `--model` take precedence. Omit `--question` for interactive use. Both providers save PNGs under `reports/` by default.

To analyze a different CSV and choose where to save the images:

```bash
python -m src.agent --csv path/to/sales.csv --output-dir reports/sales --question "Plot monthly total revenue."
```

The demo CSV supports group comparisons, histograms, and scatter plots. Time trends need a CSV with a date field such as `month` or `transaction_date`. Requests for filters and custom formulas are outside the current tool set. Each interactive question runs independently, so include any context you want the analysis to use.

## How it works

LangChain binds the analysis tools to the chat model. The LangGraph `analyst` node supplies the schema and routes tool calls to `ToolNode`. Results return to the model so it can choose another tool or summarize the analysis. A run allows up to six rounds of tool calls. Pandas calculates the statistics and Matplotlib renders the images. The CLI reads the actual file paths from the tool results.

Run tests without making LLM API calls:

```bash
python -m pytest -q
```

## Roadmap

1. **More analysis tools:** Add filtering and missing-data checks.
2. **Streamlit UI:** Upload a CSV, ask questions, view charts, and inspect tool traces.
3. **Evaluation:** Create benchmark questions with known numeric answers and track answer accuracy, tool selection, latency, and failures.

## Safety and reproducibility

- Treat CSVs as read-only inputs. Only the configured report directory receives generated files.
- The model receives the question, schema, and tool results, which can include category labels and statistics. Choose data you are authorized to share with the model provider.
- Keep credentials out of source control.
- Add dataset source/licensing information when switching from synthetic data to public data.

## References

- [LangChain Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart)
- [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [LangChain ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama)
