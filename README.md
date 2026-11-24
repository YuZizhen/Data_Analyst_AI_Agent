# Agentic Data Analyst

A CLI data analyst built with **LangChain**, **LangGraph**, **Pandas**, and **Matplotlib**. Ask a question in natural language. The agent chooses the analysis functions, summarizes their results, and returns the paths to any PNG reports it creates.

## Current functionality (MVP)

- Inspect the schema and row count of the demo data or your own CSV.
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

Copy `.env.example` to `.env`, then set `OPENAI_API_KEY` in `.env` (never commit this file). Run:

```bash
python -m src.agent
```

To run a single question and save the default dataset's reports under `reports/`:

```bash
python -m src.agent --question "Compare average resale price by town and plot the distribution of resale prices."
```

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
