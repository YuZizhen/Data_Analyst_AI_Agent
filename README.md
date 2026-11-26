# Agentic Data Analyst

A self-validating data analysis agent built with **LangChain**, **LangGraph**, **Pandas**, and **Matplotlib**. Ask a question in the browser or CLI to inspect a CSV, prepare the data, and generate downloadable PNG reports.

## What makes it different

- **Dynamic tool selection.** The model chooses inspection, cleaning, statistics, and chart tools based on the question and the current table. A single request can filter rows, compute a statistic, and create several reports.
- **Self-validation and recovery.** A LangGraph node checks the data scope, independently recomputes supported statistics, and scans the draft for common claims that exceed the evidence. It can send the analysis back for one correction within the tool budget.
- **Traceable and reproducible preparation.** The browser keeps originals and cleaned versions, shows the cleaning code, and exports a replay script. Each analysis record captures the dataset hash, tool parameters, results, and any validation attempts.

## Current functionality

- Inspect the schema, row count, and actual rows of the demo data or your own CSV.
- Profile missing values and distinct values, inspect labels, filter rows, remove missing or duplicate rows, and convert columns to number, date, or text.
- Choose an OpenAI API model or a local Ollama model with tool-calling support.
- Compare groups by mean, median, sum, or row count with a bar chart.
- Count all records in the active scope, including missing values, or list the original top or bottom records with their requested columns.
- Execute unambiguous row counts, whole-column statistics, and named grouped statistics directly. Checks cover the requested operation, selected rows, and result. Grouped answers use the tool's results, including each group's usable sample count.
- Sum user-defined row formulas by group, for example `Calculate total net by category (net = amount * (1 - discount))`. Formulas use column names, numbers, parentheses, and the basic arithmetic operators.
- Plot a metric over time, its distribution, or its relationship to another numeric metric.
- Review eligible fields against a numeric target, then rank Pearson correlations for numeric fields and correlation ratios for categorical fields separately, with paired sample sizes.
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

Open `http://127.0.0.1:8501`. The sidebar lets you upload a CSV, inspect its fields, and choose Ollama or OpenAI. The demo data and local `qwen3.5:4b` are selected initially unless you have set model defaults. In Ollama mode, enter the exact tag of an installed model that supports tool calling. In OpenAI mode, enter an API key or set `OPENAI_API_KEY` in your environment. Try an example question or type one below the conversation. Each report can be downloaded as a PNG from its message.

Open **预览数据** above the chat to inspect the first ten rows without asking a model. You can also ask to see the current dataset in the chat. **查看执行记录** below a tool-backed reply shows the tools that returned results or errors and offers a JSON download.

The execution record opens to **分析旅程**, which explains the purpose, parameters, and recorded results of each tool step in Chinese. **校验结果** shows the checks and outcomes for numerical analysis when validation runs. **原始工具数据** preserves the exact tool parameters and outputs. The JSON download includes the tool calls, the journey, and any validation attempts.

While an answer is running, an animated analyst types at a workstation and the current action appears in a thought bubble. Each bubble describes the selected analysis, including its fields and parameters. Meaningful steps move into the conversation history, while the initial waiting cue disappears. Browser reduced-motion preferences disable the animations.

The sidebar's **当前使用的数据表** selector lists uploaded originals and explicitly saved versions. Filters and cleaning affect the current conversation's working table and are available for follow-up questions. **清除临时范围** returns to the selected table, and a new conversation clears the temporary scope. To keep a version, use **保存当前范围为数据版本** or ask **保存当前数据表** in chat. Saved row-only selections are labeled **筛选版**, while versions with cleaning steps are labeled **清洗版**. The originals and saved versions are stored locally as Parquet under `data/private/datasets/`, which is excluded from Git. The **恢复原始数据** button selects the original table. The last four exchanges provide chat context within a conversation.

For a saved version, open **筛选逻辑与代码** or **清洗逻辑与代码** to see the operations, parameters, row counts, replay script, Python implementations, and CSV download. Run the replay script from the project environment. It writes a separate output file.

### Try it with the demo CSV

Use the original demo table for each example. Clear the temporary scope after the filtering example to return to all rows.

| Ask in the chat | What to look for |
| --- | --- |
| 我可以看一下这个数据库吗？ | A preview of the active table in the chat. |
| 只看 `town = BISHAN`，按 `town` 画 `resale_price` 的均价图。 | A temporary filtered scope, a chart, and a validation record showing how many rows matched and the independently recomputed mean. |
| 哪些字段与 `resale_price` 的关联最强？ | Candidate review followed by separate numeric and categorical rankings, with sample sizes and a partial validation status. |

### Command-line interface

#### Local Ollama

Start Ollama and check that the model you want appears in `ollama list`. The agent defaults to `qwen3.5:4b` in Ollama mode. To choose any installed model that supports tool calling, pass its exact tag to `--model`.

```bash
python -m src.agent --provider ollama --model qwen3.5:4b --question "Compare average resale price by town and plot the distribution of resale prices."
```

This mode needs no OpenAI API key. It connects to `http://127.0.0.1:11434` by default. Set `OLLAMA_BASE_URL` in `.env` if your Ollama server runs at a different address. The agent requests an 8,192-token Ollama context for tool definitions, chat history, and replies. `OLLAMA_NUM_CTX` overrides it. Larger contexts use more memory. Ollama manages model storage through its own settings, including `OLLAMA_MODELS`.

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

### Regression and feature importance

Ask “用一棵简单的决策树预测 `RESALE_PRICE`，75% 训练、25% 测试” to evaluate a model. Choose a single shallow decision tree, linear regression, ridge regression, or random forest. The default evaluation uses a random 80/20 split. You can request a chronological split with a date field, or three-fold cross-validation. A chronological split keeps records sharing a date together and reports the actual counts and boundary. The execution record includes model parameters, features, train/test counts, MAE, median-baseline MAE, RMSE, and R². Trainable preprocessing, including imputation and categorical encoding, uses only the training subset.

For an individual estimate, give the user-known inputs, for example “用决策树估价 `FLOOR_AREA_SQM` 为 90、`FLAT_TYPE` 为 `4 ROOM` 的房源，目标是 `RESALE_PRICE`”。 The estimate uses only the specified eligible columns and includes held-out error. The language model can inspect an unfamiliar CSV's schema and categories to map the question to tools. Supported estimators are defined in the project. The tools use up to 12,000 usable rows, sampled with a fixed seed for larger tables, and retain the original and sampled counts in the execution record.

A feature-importance request uses a separate report, with training and permutation progress shown in the browser. When asking for a chart after a field association ranking, the browser uses a scatter plot and a grouped box plot for the leading eligible fields on the same working table.

Ask for the top N factors predicting a numeric column and their shares. The tool trains a random forest with three shuffled validation folds, shuffles each original factor independently, and measures the change in mean absolute error (MAE). The share divides each positive mean increase by the total across all positive factors. The chart shows the top N and groups the remaining positive shares under **Other factors**. Shares describe how this model uses the supplied data. Correlated factors can split or mask importance. Use only candidate fields available when making the prediction.

The tool needs at least 30 working rows with valid targets. It reports validation MAE against a median baseline and counts how many folds beat the baseline. The six-row HDB demo supports descriptive relationship rankings. Use a larger CSV for predictive importance. The browser's **因素重要性怎么解读** panel shows the baseline comparison, the percentage shares, and how stable the rankings were across the folds. These scores describe predictive performance on the supplied data. Establishing causation or generalization requires additional evidence.

## Candidate field review

Before ranking associations, evaluating predictions, or computing feature importance, the agent reviews candidates against the chosen target on the active table. It records how many rows have values in both fields, how many different values remain, and whether each candidate was **入选**, **跳过**, or **待确认**. Constant fields, too few usable rows, and undersized groups for categorical association are examples of reasons to skip a field. High missingness is recorded alongside the decision.

Identifier-like names and exact copies of the target are held for confirmation when otherwise eligible. Predictive importance also screens for common names that suggest the field becomes available after the outcome. These signals are heuristic and only cover some kinds of leakage. In the browser, open **查看执行记录 → 候选字段审查** for the reasons and row counts. The same evidence is in the JSON download. To include one of the preceding analysis's pending fields, send a line such as ``确认纳入 `order_id` ``. The agent reruns the tool with the same target and candidate scope. For a standalone CLI question, put ``Confirm inclusion of `order_id` `` on a separate line within the full request.

## Validation and recovery

When validation runs, the progress bubble shows **正在校验**. Open **查看执行记录 → 校验结果** to see what the agent checked, the evidence or recomputed values, and any retry. The answer displays **校验通过**, **部分校验**, or **校验未通过**.

| Check | Current coverage |
| --- | --- |
| Data scope | Records requested conditions separately from executed filters. Replays cleaning and compares the actual rows with the requested scope. Explicit comparisons can be matched directly. The model also reads the whole conversational request and maps category filters to existing values in the active data. The record shows the phrase and matched value. A clause that cannot be resolved is marked as partial coverage. |
| Record requests | Counts all rows in the current scope or ranks original records by a numeric field. The execution record shows the row count or the fields and records selected. A request for text or a table suppresses chart tool calls. |
| Group sizes | Grouped summaries show each group's statistic and the number of valid rows behind it. Those counts are independently checked and displayed from the tool output. |
| Independent recalculation | Recomputes grouped statistics, pairwise Pearson correlations, and the median and range of distributions. Trends, field association rankings, and feature importance have partial coverage. |
| Conclusion wording | Scans the draft for common unsupported causal or statistical significance claims. This is a wording check. Cross-checking every number stated in the answer is on the backlog. |

If a check fails, the agent can retry once within the tool-call budget. Rejected charts are excluded from the answer, and the execution record keeps the findings from each attempt.

## How it works

LangChain binds the analysis tools to the chat model. LangGraph routes simple browse requests straight to the preview tool. For analysis requests, the `analyst` node supplies the working schema and routes tool calls to `ToolNode`. Preparation tools update the active table in graph state before dependent calls run. The `validate` node checks numerical analysis before the reply is displayed and can route back to `analyst` for recovery. A run allows up to eight tool-call rounds. Pandas computes statistics and Matplotlib renders images.

Tool modules live under `src/tools/`. `dataset.py` defines the shared data context, `inspection.py` shows rows and schema, `preprocessing.py` handles preparation, `charts.py` builds charts, `relationships.py` handles association analysis, and `importance.py` produces predictive importance reports.

Run tests without making LLM API calls:

```bash
python -m pytest -q
```

## Backlog

1. Extend independent recalculation to relationship rankings, trends, and feature importance. Expand scope checks to complex conditions such as alternatives and relative dates, and compare stated numerical claims with structured results.
2. Replay recorded tool calls against a CSV with the matching SHA-256 hash, without calling a model, and compare the results.
3. Add regression cases for scope selection, numerical accuracy, and conclusions that overstate the evidence.

## Safety and reproducibility

- Treat CSVs as read-only inputs. The CLI saves images to the configured report directory. The browser interface prepares downloads using temporary files.
- The model receives the question, schema, and tool results, which can include category labels and statistics. Choose data you are authorized to share with the model provider.
- Keep credentials out of source control.
- Add dataset source/licensing information when switching from synthetic data to public data.

## References

- [LangChain Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart)
- [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [LangChain ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama)
