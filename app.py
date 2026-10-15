import io

import pandas as pd
import streamlit as st

from langchain_agent import DataAgent

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

if "agent" not in st.session_state:
    st.session_state.agent = DataAgent(st.secrets["HF_key"])
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

agent: DataAgent = st.session_state.agent

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
    st.subheader("🧠 Memory")
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


def show_steps(steps):
    if not steps:
        return
    failed = sum(1 for s in steps if not s.ok)
    label = f"🔧 Agent steps ({len(steps)}" + (f", {failed} fixed" if failed else "") + ")"
    with st.expander(label):
        for i, s in enumerate(steps, 1):
            icon = "✅" if s.ok else "⚠️"
            st.markdown(f"**{i}. {icon} {s.tool}**")
            if s.tool == "inspect_data":
                if s.code:
                    st.caption(f"column: {s.code}")
            else:
                st.code(s.code, language="python")
            st.text(s.output[:1500])


# ---------------------------------------------------------------- main chat

if df is None:
    st.info("Upload a `.csv` or `.xlsx` file, or pick a sample dataset.")
    st.stop()

st.dataframe(df.head(), width="stretch")
st.markdown("### 💬 Ask a question")

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

if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Working on it..."):
            try:
                reply = agent.ask(prompt, {"data": df})
                msg = {"role": "assistant", "content": reply.answer,
                       "charts": reply.charts, "steps": reply.steps}
                if reply.data_changed:
                    new_df = reply.datasets["data"]
                    st.session_state.df = new_df
                    msg["data_note"] = (f"✅ Dataset updated: {df.shape[0]}×{df.shape[1]} → "
                                        f"{new_df.shape[0]}×{new_df.shape[1]}")
            except Exception as e:
                msg = {"role": "assistant", "content": f"⚠️ Something went wrong: {e}"}

    # Save first, then rerun so the chat, sidebar and preview all show the new state.
    st.session_state.chat_messages.append(msg)
    st.rerun()
