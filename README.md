# AI Data Analysis Agent

Ask questions about a CSV or Excel file in plain English. The agent looks at the data, runs calculations, draws charts, cleans data when asked, and fixes its own code when something fails.

## How it works

The LLM (Groq, Hugging Face router or Gemini — picked in the sidebar) runs in a loop with four tools:

| Tool | What it does |
|---|---|
| `inspect_data` | Shows columns, types, nulls, sample rows, or details for one column |
| `run_analysis` | Runs pandas or DuckDB SQL code and reads the printed result |
| `make_chart` | Builds a Plotly / Matplotlib / Altair chart and checks it isn't empty |
| `update_dataset` | Cleans or transforms the data; later steps use the new version |
| `explain_change` | Root-cause finder: breaks a metric's change between two periods/groups down by every column and ranks the segments that drove it |

If code fails, the error goes back to the agent so it can fix it and try again (up to 4 failed attempts per question). Every step is shown live in the app under **Agent steps**.

- **Critic check** (sidebar toggle): a second LLM call reviews each answer — does the code match the question, do the numbers match the tool outputs, is anything overclaimed? If not, the agent fixes it once.
- **📋 Analyze this dataset**: the agent explores on its own and writes a short report with 3 charts, key insights and a root-cause breakdown.

## Files

- `app.py` – Streamlit interface
- `langchain_agent.py` – the agent, its tools and the retry loop
- `AgentClass.py` – safety checks and code runner
- `rootcause.py` – the root-cause breakdown (plain pandas)

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put your settings in `.streamlit/secrets.toml` (fill in at least one key):

```toml
PROVIDER = "groq"               # "groq", "hf" or "gemini"
MODEL = "openai/gpt-oss-120b"   # default model for that provider

GROQ_key = "gsk_..."
HF_key = ""
GEMINI_key = ""
```

## Safety

Agent-written code is checked before it runs: no imports, no file or OS access, no internal (`_`) attributes, and DuckDB has file access switched off. This blocks the common escapes, but in-process Python is never a perfect sandbox, so if you host the app publicly, run it in a container with no secrets on disk.
