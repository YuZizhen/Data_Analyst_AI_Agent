# Agentic Data Analyst

A data analyst built with **LangChain**, **LangGraph**, **Pandas**, and **Matplotlib**, with a local browser interface and a CLI. Ask a question in natural language. The agent chooses the analysis functions, summarizes their results, and generates PNG reports.

## Current functionality (MVP)

- Inspect the schema, row count, and actual rows of the demo data or your own CSV.
- Profile missing values and distinct values, inspect labels, filter rows, remove missing or duplicate rows, and convert columns to number, date, or text.
- Choose an OpenAI API model or a local Ollama model with tool-calling support.
- Compare groups by mean, median, sum, or row count with a bar chart.
- Plot a metric over time, its distribution, or its relationship to another numeric metric.
- Compare all candidate fields against a numeric target by ranking numeric Pearson correlations and categorical correlation ratios separately, with paired sample sizes.
- Plot the top N predictive factors for a numeric target with percentage shares and an Other factors bar when needed.
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

The execution record opens to **分析旅程**, which explains the purpose, parameters, and recorded results of each tool step in Chinese. The **原始工具数据** tab preserves the exact tool parameters and outputs. Both views come from the recorded calls and are included in the JSON download.

While an answer is running, an animated analyst types at a workstation and the current action appears in a thought bubble. Each bubble describes the selected analysis, including its fields and parameters. Meaningful steps move into the conversation history, while the initial waiting cue disappears. Browser reduced-motion preferences disable the animations.

The sidebar's **当前使用的数据表** selector lists uploaded originals and saved cleaned versions. Successful preparation creates or reuses a version and selects it for follow-up questions. The originals and versions are stored locally as Parquet under `data/private/datasets/`, which is excluded from Git. Manual table or model-provider changes start a new conversation. The **恢复原始数据** button selects the original table. The last four exchanges provide chat context within a conversation.

For a cleaned version, open **清洗逻辑与代码** to see each operation, its parameters and row counts, a replay script for the saved original, the actual Python implementations, and a CSV download. Run the replay script from the project environment. It writes a separate output file.

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

The demo CSV supports group comparisons, histograms, and scatter plots. Time trends need a date column. The interactive CLI retains the prepared data until you ask to restore the original dataset or exit. Include any other context you want the analysis to use in each question.

### Data preparation

Upload a UTF-8 CSV from any domain, including a CSV exported from a database. Use the file's actual field names or describe the scope in natural language. For example, ask “Use only rows where the market is Singapore, then compare average unit price by product category.” Text filters ignore case and surrounding whitespace. They support exact matches, exclusions, lists, and literal substring matches. Ordered comparisons use numbers or a column already converted to dates. For non-ISO dates, give the date format explicitly.

Tools can also profile the data, inspect common values, drop rows missing selected fields, deduplicate by selected fields or entire rows, and convert numbers, dates, or text. Failed conversions are counted, and an empty filter remains empty. The original CSV stays read-only. Execution records include preparation operations, parameters, and before-and-after row counts. Direct connections to live databases would require a separate data source adapter.

### Feature importance

Ask for the top N factors predicting a numeric column and their shares. The tool trains a random forest with three shuffled validation folds, shuffles each original factor independently, and measures the change in mean absolute error (MAE). The share divides each positive mean increase by the total across all positive factors. The chart shows the top N and groups the remaining positive shares under **Other factors**. Shares describe how this model uses the supplied data. Correlated factors can split or mask importance. Use only candidate fields available when making the prediction.

The tool needs at least 30 working rows with valid targets and checks that validation MAE beats a median baseline. The six-row HDB demo supports descriptive relationship rankings. It is too small for a predictive importance report. The browser's evidence panel and the model's explanation use the held-out MAE comparison and per-fold stability counts. These counts describe this validation setup, and further checks are needed to establish causation or performance on other data.

## How it works

LangChain binds the analysis tools to the chat model. LangGraph routes simple browse requests straight to the preview tool. For analysis requests, the `analyst` node supplies the working schema and routes tool calls to `ToolNode`. Preparation tools update the working table in graph state. Dependent analysis calls run after that update. A run allows up to eight tool-call rounds. Pandas computes statistics and Matplotlib renders images.

Before displaying a numerical analysis, the `validate` node replays recorded preparation and compares the reconstructed data with the tool input. It checks explicit filters like `market = Singapore` and independently recomputes group summaries, pairwise Pearson correlations, and distribution summaries. It scans the draft for common unsupported causal or statistical-significance claims. The browser shows the current check in the activity bubble and saves each attempt under **查看执行记录 → 校验结果** and in the downloadable JSON. A failed check can route back to the analyst once within the tool budget. Rejected charts are excluded from the answer. Other statistical tools can show partial coverage while independent recalculation is pending.

Tool modules live under `src/tools/`. `dataset.py` defines the shared data context, `inspection.py` shows rows and schema, `preprocessing.py` handles preparation, `charts.py` builds charts, `relationships.py` handles association analysis, and `importance.py` produces predictive importance reports.

Run tests without making LLM API calls:

```bash
python -m pytest -q
```

## Backlog

1. Extend independent recalculation to relationship rankings, trends, and feature importance. Expand scope checks beyond explicit field equality and compare stated numerical claims with structured results.
2. Review the suitability of analysis candidates before ranking, including numeric identifiers, constant columns, missing values, and usable sample sizes.
3. Replay recorded tool calls against a CSV with the matching SHA-256 hash, without calling a model, and compare the results.
4. Add regression cases for scope selection, numerical accuracy, and conclusions that overstate the evidence.

## Safety and reproducibility

- Treat CSVs as read-only inputs. The CLI saves images to the configured report directory. The browser interface prepares downloads using temporary files.
- The model receives the question, schema, and tool results, which can include category labels and statistics. Choose data you are authorized to share with the model provider.
- Keep credentials out of source control.
- Add dataset source/licensing information when switching from synthetic data to public data.

## References

- [LangChain Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart)
- [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [LangChain ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama)
