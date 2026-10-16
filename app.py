import io
import threading
import time

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from langchain_agent import AUTO_REPORT_LABEL, AUTO_REPORT_PROMPT, DEFAULT_PROVIDER, PROVIDERS, DataAgent
from memory import DEFAULT_PATH as MEMORY_PATH
from memory import RuleMemory, gemini_embedder, parse_remember
from report import build_report

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide", page_icon="📊")

st.title("📊 AI Data Analysis Agent")
st.markdown(
    """
Upload a **CSV or Excel** file (or pick a sample) and ask questions in plain English.
The agent looks at your data, runs calculations, draws charts, and fixes its own mistakes.

- "Which cancer type has the highest average risk score? Show it as a bar chart."
- "Create a stacked area chart of Movies and TV Shows released each year."
- "Drop rows where Booking Value is missing, then show average value by Vehicle Type."
"""
)

SAMPLES = {
    "Cancer risk factors": "cancer-risk-factors.csv",
    "Netflix titles": "netflix_titles.csv",
    "Uber bookings": "ncr_ride_bookings.csv",
}


@st.cache_data
def load_sample(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def set_data(df: pd.DataFrame, label: str):
    st.session_state.df = df
    st.session_state.df_original = df
    st.session_state.data_label = label


# ---------------------------------------------------------------- state

# Provider + model: PROVIDER / MODEL in secrets.toml set the defaults; the sidebar can change them.
# Only providers whose API key is filled in are offered.
providers = [p for p, cfg in PROVIDERS.items() if st.secrets.get(cfg["key"], "").strip()]
if not providers:
    st.error("No API key found. Add GROQ_key, HF_key or GEMINI_key to .streamlit/secrets.toml.")
    st.stop()
default_provider = st.secrets.get("PROVIDER", DEFAULT_PROVIDER)
provider = st.sidebar.selectbox("🔌 Provider", providers,
                                index=providers.index(default_provider) if default_provider in providers else 0)

model_options = list(PROVIDERS[provider]["models"])
secret_model = st.secrets.get("MODEL", "")
if provider == default_provider and secret_model:
    if secret_model not in model_options:
        model_options.insert(0, secret_model)
    model_index = model_options.index(secret_model)
else:
    model_index = 0
model = st.sidebar.selectbox("🤖 Model", model_options, index=model_index)
use_critic = st.sidebar.toggle("🧐 Critic check", value=True,
                               help="A second LLM call reviews each answer (code, numbers, claims) "
                                    "and the agent fixes problems once. Slower, but more reliable.")

current = st.session_state.get("agent")
if current is None or (current.provider, current.model) != (provider, model):
    old = current
    st.session_state.agent = DataAgent(st.secrets[PROVIDERS[provider]["key"]], model=model, provider=provider)
    if old is not None:
        st.session_state.agent.history = old.history  # keep chat memory when switching
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Learning memory. Rules are saved to disk only with SAVE_MEMORY = true (local use); on the
# public app each visitor keeps their own rules for the session, so nobody can change others' answers.
if "rule_memory" not in st.session_state:
    gemini_key = st.secrets.get("GEMINI_key", "").strip()
    st.session_state.rule_memory = RuleMemory(
        path=MEMORY_PATH if st.secrets.get("SAVE_MEMORY", False) else None,
        embed=gemini_embedder(gemini_key) if gemini_key else None)

agent: DataAgent = st.session_state.agent
rule_memory: RuleMemory = st.session_state.rule_memory

# ---------------------------------------------------------------- data loading
# Data is only (re)loaded when the user picks a *different* sample or uploads a
# *different* file, so cleaned data isn't overwritten on every rerun.

sample_choice = st.selectbox("Try a sample dataset:", ["None"] + list(SAMPLES))
if sample_choice != st.session_state.get("last_sample", "None"):
    st.session_state.last_sample = sample_choice
    if sample_choice != "None":
        set_data(load_sample(SAMPLES[sample_choice]), f"sample: {sample_choice}")

st.markdown("### 📎 Upload your dataset")
uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
if uploaded is not None:
    upload_id = (uploaded.name, uploaded.size)
    if upload_id != st.session_state.get("last_upload"):
        st.session_state.last_upload = upload_id
        try:
            file_io = io.BytesIO(uploaded.getvalue())
            if uploaded.name.lower().endswith(".csv"):
                new_df = pd.read_csv(file_io)
            else:
                new_df = pd.read_excel(file_io, engine="openpyxl")
            set_data(new_df, f"file: {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

df = st.session_state.get("df")

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🧹 Clear conversation", width="stretch"):
        st.session_state.chat_messages = []
        agent.clear()
        st.rerun()

    if st.session_state.chat_messages:
        report_html = build_report(st.session_state.chat_messages, {
            "Dataset": st.session_state.get("data_label", ""),
            "Size": f"{df.shape[0]:,} rows × {df.shape[1]} columns" if df is not None else "",
            "Model": f"{agent.provider} / {agent.model}",
        })
        st.download_button("📄 Download report (HTML)", report_html,
                           f"analysis_report_{time.strftime('%Y%m%d_%H%M')}.html", "text/html",
                           width="stretch", help="All questions, answers, charts and agent steps in one file.")

    if df is not None and df is not st.session_state.get("df_original"):
        if st.button("↩️ Undo all data changes", width="stretch"):
            st.session_state.df = st.session_state.df_original
            st.rerun()

    st.markdown("---")
    if df is not None:
        st.subheader("📊 Dataset overview")
        st.caption(st.session_state.get("data_label", ""))
        st.write(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")
        with st.expander("📝 Column details"):
            for col in df.columns:
                st.write(f"- **{col}** ({df[col].dtype}) | {int(df[col].isna().sum())} nulls")

    st.markdown("---")
    st.subheader("🧠 Saved rules")
    st.caption("Rules and definitions the agent follows when relevant. "
               "You can also type `remember: ...` in the chat.")
    with st.form("add_rule", clear_on_submit=True, border=False):
        new_rule = st.text_input("New rule", placeholder="e.g. Revenue = sum of Booking Value",
                                 label_visibility="collapsed")
        if st.form_submit_button("💾 Save rule", width="stretch") and new_rule.strip():
            rule_memory.add(new_rule)
            st.rerun()
    for rule in rule_memory.rules:
        text_col, del_col = st.columns([5, 1])
        text_col.write(f"• {rule['text']}")
        if del_col.button("🗑", key=f"del_{rule['id']}", help="Delete this rule"):
            rule_memory.delete(rule["id"])
            st.rerun()
    if not rule_memory.path:
        st.caption("Rules last for this session only.")

    st.markdown("---")
    st.subheader("💬 Recent chat")
    if not agent.history:
        st.caption("Nothing yet.")
    for msg in agent.history[-4:]:
        role = "User" if msg.type == "human" else "Agent"
        st.write(f"**{role}:** {msg.content[:300]}")

# ---------------------------------------------------------------- chat helpers


def show_chart(chart, key: str):
    if chart.kind == "plotly":
        st.plotly_chart(chart.obj, key=f"plot_{key}")
        st.download_button("🌐 Download HTML", chart.obj.to_html(), f"chart_{key}.html",
                           "text/html", key=f"dl_{key}")
    elif chart.kind == "matplotlib":
        st.pyplot(chart.obj)
        buf = io.BytesIO()
        chart.obj.savefig(buf, format="png", bbox_inches="tight")
        st.download_button("💾 Download PNG", buf.getvalue(), f"chart_{key}.png",
                           "image/png", key=f"dl_{key}")
    elif chart.kind == "altair":
        st.altair_chart(chart.obj, key=f"plot_{key}")


def steps_label(steps) -> str:
    failed = sum(1 for s in steps if not s.ok)
    return f"🔧 Agent steps ({len(steps)}" + (f", {failed} fixed" if failed else "") + ")"


def show_step(i: int, s):
    icon = "✅" if s.ok else "⚠️"
    if s.tool == "critic":
        st.markdown(f"**{i}. 🧐 critic: {'approved' if s.ok else 'found problems'}**")
        (st.success if s.ok else st.warning)(s.output)
        return
    if s.tool == "memory":
        st.markdown(f"**{i}. 🧠 memory**")
        st.info(s.output)
        return
    st.markdown(f"**{i}. {icon} {s.tool}**")
    if s.tool == "inspect_data":
        if s.code:
            st.caption(f"column: {s.code}")
    else:
        st.code(s.code, language="python")
    st.text(s.output[:1500])


def show_steps(steps):
    if not steps:
        return
    with st.expander(steps_label(steps)):
        for i, s in enumerate(steps, 1):
            show_step(i, s)


# ---------------------------------------------------------------- main chat

if df is None:
    st.info("Upload a `.csv` or `.xlsx` file, or pick a sample dataset.")
    st.stop()

st.dataframe(df.head(), width="stretch")
st.markdown("### 💬 Ask a question")
if st.button(AUTO_REPORT_LABEL, type="primary",
             help="The agent explores the data on its own: key facts, 3 charts, "
                  "what drives the main change, and a short report."):
    st.session_state.pending_prompt = AUTO_REPORT_LABEL
    st.rerun()

for i, msg in enumerate(st.session_state.chat_messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            show_steps(msg.get("steps"))
        st.markdown(msg["content"])
        if msg.get("data_note"):
            st.success(msg["data_note"])
        for j, chart in enumerate(msg.get("charts", [])):
            show_chart(chart, f"{i}_{j}")

if not st.session_state.chat_messages:
    st.markdown("##### 💡 Suggested questions")
    suggestions = [
        "Give me a quick overview of this dataset",
        "What are the most interesting patterns? Show one chart.",
        "Are there missing values? Clean them up.",
    ]
    for col, s in zip(st.columns(len(suggestions)), suggestions):
        if col.button(s, width="stretch"):
            st.session_state.pending_prompt = s
            st.rerun()

prompt = st.chat_input("Ask something about the data...")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt and (rule_text := parse_remember(prompt)):
    # "remember: ..." saves a rule; no agent call needed.
    rule_memory.add(rule_text)
    st.session_state.chat_messages += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"💾 Saved rule: *{rule_text}*. I'll follow it in related questions."},
    ]
    st.rerun()

if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Live steps: each tool call is added to this box as soon as it finishes.
        status = st.status("🧠 Working on it...", expanded=True)
        live_steps = []
        ctx = get_script_run_ctx()

        def on_step(step):
            # Tools run in a LangGraph worker thread; attach it to this session so st calls work.
            add_script_run_ctx(threading.current_thread(), ctx)
            live_steps.append(step)
            status.update(label=f"🧠 Working... step {len(live_steps)}: {step.tool}")
            with status:
                show_step(len(live_steps), step)

        try:
            question = AUTO_REPORT_PROMPT if prompt == AUTO_REPORT_LABEL else prompt
            rules = [r["text"] for r, _ in rule_memory.relevant(question)]
            reply = agent.ask(question, {"data": df}, on_step=on_step, critic=use_critic, rules=rules)
            status.update(label=steps_label(reply.steps) if reply.steps else "✅ Done",
                          state="complete", expanded=False)
            msg = {"role": "assistant", "content": reply.answer,
                   "charts": reply.charts, "steps": reply.steps}
            if reply.data_changed:
                new_df = reply.datasets["data"]
                st.session_state.df = new_df
                msg["data_note"] = (f"✅ Dataset updated: {df.shape[0]}×{df.shape[1]} → "
                                    f"{new_df.shape[0]}×{new_df.shape[1]}")
        except Exception as e:
            status.update(label="⚠️ Stopped with an error", state="error", expanded=False)
            msg = {"role": "assistant", "content": f"⚠️ Something went wrong: {e}",
                   "steps": live_steps}

    # Save first, then rerun so the chat, sidebar and preview all show the new state.
    st.session_state.chat_messages.append(msg)
    st.rerun()
