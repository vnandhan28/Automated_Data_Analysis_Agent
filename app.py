import io
import keyword
import re
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from AgentClass import RESERVED_NAMES
from langchain_agent import AUTO_REPORT_LABEL, AUTO_REPORT_PROMPT, DEFAULT_PROVIDER, PROVIDERS, DataAgent
from memory import DEFAULT_PATH as MEMORY_PATH
from memory import RuleMemory, gemini_embedder, parse_remember
from report import build_report
from theme import APP_CSS, NOTE_STEPS, STEPS, highlight_markdown, step_meta

st.set_page_config(page_title="Ask your data", layout="wide", page_icon="📊")
st.html(APP_CSS)

# Sample datasets shown as cards. "uber_prices" has no card of its own: the Uber card offers it.
SAMPLE_SETS = {
    "cancer": dict(name="Cancer risk factors", files=["cancer-risk-factors.csv"], image="icons/cancer.png",
                   desc="Patients with lifestyle and genetic risk factors and an overall risk score."),
    "netflix": dict(name="Netflix titles", files=["netflix_titles.csv"], image="icons/netflix.png",
                    desc="Movies and TV shows with ratings, countries and release years."),
    "uber": dict(name="Uber ride bookings", files=["ncr_ride_bookings.csv"], image="icons/uber.png",
                 desc="A year of rides in Delhi NCR: status, vehicle type, value and ratings."),
    "uber_prices": dict(name="Uber bookings + vehicle prices", files=["ncr_ride_bookings.csv", "vehicle_types.csv"]),
}
AVATARS = {"user": ":material/person:", "assistant": ":material/insights:"}
EXAMPLES = {  # example question -> sample it needs
    "Which cancer type has the highest average risk score?": "cancer",
    "How many movies and TV shows came out each year?": "netflix",
    "Why did booking value change from March to April 2024?": "uber",
}


@st.cache_data
def load_sample(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def dataset_name(filename: str, taken) -> str:
    """'cancer-risk-factors.csv' -> 'cancer_risk_factors' (a valid, unused Python name)."""
    base = re.sub(r"\W+", "_", Path(filename).stem.lower()).strip("_") or "data"
    if base[0].isdigit() or keyword.iskeyword(base) or base in RESERVED_NAMES:
        base = f"t_{base}"
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    return name


METRIC_HINTS = ("value", "amount", "price", "revenue", "sales", "score", "total", "cost", "fare", "profit",
                "income", "rating", "spend", "count", "quantity")
GROUP_HINTS = ("type", "category", "segment", "region", "class", "group", "status", "country", "method")


def suggest_questions(datasets) -> list:
    """Starter questions built from the first table's columns (works for any uploaded file, no AI call)."""
    df = next(iter(datasets.values()))
    cols = [str(c) for c in df.columns]
    year_col = next((c for c in cols if "year" in c.lower() and pd.api.types.is_numeric_dtype(df[c])), None)
    date_col = next((c for c in cols if "date" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])), None)
    def looks_like_id(c):  # unique whole numbers (order_id 1, 2, 3...); unique decimals are measurements
        s = df[c].dropna()
        return s.is_unique and len(s) > 50 and bool((s % 1 == 0).all())

    def hint_rank(c, hints):  # earlier hints win ("type" before "status"); no hint ranks last
        return next((i for i, h in enumerate(hints) if h in c.lower()), len(hints))

    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
               and df[c].nunique() > 10 and c != year_col and not looks_like_id(c)]
    groups = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= 20]
    metric = min(numeric, key=lambda c: (hint_rank(c, METRIC_HINTS), -df[c].notna().mean()), default=None)
    group = min(groups, key=lambda c: (hint_rank(c, GROUP_HINTS), df[c].isna().mean(), abs(df[c].nunique() - 6)),
                default=None)
    qs = []
    if len(datasets) > 1:
        qs.append("How can these tables be joined? Join them and summarize the result.")
    if metric and group:
        qs.append(f"What is the average {metric} by {group}? Show a bar chart.")
    if metric and date_col:
        qs.append(f"Is total {metric} going up or down over time?")
    elif year_col:
        qs.append(f"How many rows are there per {year_col}? Show it as a line chart.")
    elif group:
        qs.append(f"Which {group} is most common, and what share of rows is it?")
    if metric and len(numeric) >= 2:
        qs.append(f"What is most strongly related to {metric}?")
    qs.append("Are there missing values or duplicates? Clean them up.")
    if len(qs) < 3:
        qs.insert(0, "Give me a quick overview of this dataset")
    return qs[:4]


def md_escape(text: str) -> str:
    """Show tool output as plain text inside markdown."""
    return re.sub(r"([\\`*_\[\]<>#|~$])", r"\\\1", text)


# Named datasets. "datasets" are the working copies the agent may change; "originals" are
# what was loaded (for undo); "sources" says where each came from: "sample", "upload:<file id>"
# or "agent" (a table the agent created, e.g. a join).
for key in ("datasets", "originals", "sources"):
    st.session_state.setdefault(key, {})


def add_dataset(name: str, df: pd.DataFrame, source: str):
    st.session_state.datasets[name] = df
    st.session_state.originals[name] = df
    st.session_state.sources[name] = source


def remove_datasets(predicate):
    for name in [n for n, src in st.session_state.sources.items() if predicate(src)]:
        for key in ("datasets", "originals", "sources"):
            st.session_state[key].pop(name, None)


def use_sample(sample_key: str):
    """Replace the current sample data (uploads stay) with one of SAMPLE_SETS."""
    remove_datasets(lambda src: src == "sample")
    for path in SAMPLE_SETS[sample_key]["files"]:
        add_dataset(dataset_name(path, st.session_state.datasets), load_sample(path), "sample")
    st.session_state.last_sample = sample_key


def ask(question: str):
    st.session_state.pending_prompt = question


def run_example():
    question = st.session_state.get("example_chip")
    if question:
        use_sample(EXAMPLES[question])
        ask(question)
    st.session_state.example_chip = None


def run_suggestion():
    if question := st.session_state.get("suggestion_chip"):
        ask(question)
    st.session_state.suggestion_chip = None


# ---------------------------------------------------------------- sidebar: model and checks

providers = [p for p, cfg in PROVIDERS.items() if st.secrets.get(cfg["key"], "").strip()]
if not providers:
    st.title("Ask your data a question")
    st.error("Add an API key to start: put GROQ_key or GEMINI_key in .streamlit/secrets.toml, then reload.")
    st.stop()

with st.sidebar:
    st.subheader("🧠 Model")
    st.caption("The AI model that answers your questions.")
    default_provider = st.secrets.get("PROVIDER", DEFAULT_PROVIDER)
    provider = st.selectbox("Provider", providers,
                            index=providers.index(default_provider) if default_provider in providers else 0)
    model_options = list(PROVIDERS[provider]["models"])
    secret_model = st.secrets.get("MODEL", "")
    if provider == default_provider and secret_model:
        if secret_model not in model_options:
            model_options.insert(0, secret_model)
        model_index = model_options.index(secret_model)
    else:
        model_index = 0
    model = st.selectbox("Model", model_options, index=model_index)

    st.subheader("🛡️ Checks")
    st.caption("Extra steps that make answers more reliable but slower.")
    use_critic = st.toggle("Critic check", value=True,
                           help="A second AI call reviews each answer (code, numbers, claims); "
                                "the agent fixes problems once.")
    use_plan = st.toggle("Plan first", value=True,
                         help="For bigger, multi-part requests the agent writes a short plan first. "
                              "Simple questions skip this.")

current = st.session_state.get("agent")
if current is None or (current.provider, current.model) != (provider, model):
    old = current
    st.session_state.agent = DataAgent(st.secrets[PROVIDERS[provider]["key"]], model=model, provider=provider)
    if old is not None:
        st.session_state.agent.history = old.history  # keep chat memory when switching
st.session_state.setdefault("chat_messages", [])

# Learning memory. Rules are saved to disk only with SAVE_MEMORY = true (local use); on the
# public app each visitor keeps their own rules for the session, so nobody can change others' answers.
if "rule_memory" not in st.session_state:
    gemini_key = st.secrets.get("GEMINI_key", "").strip()
    st.session_state.rule_memory = RuleMemory(
        path=MEMORY_PATH if st.secrets.get("SAVE_MEMORY", False) else None,
        embed=gemini_embedder(gemini_key) if gemini_key else None)

agent: DataAgent = st.session_state.agent
rule_memory: RuleMemory = st.session_state.rule_memory

# ---------------------------------------------------------------- hero

st.title("Ask your data a question")
st.markdown(":gray[Upload a spreadsheet and get the answer, the chart, the code behind it, and a second check.]")
st.caption(f"Model: {model} · Critic check {'on' if use_critic else 'off'}")

# ---------------------------------------------------------------- start: samples, upload, examples
# Shown open until data is loaded, then folded into an expander. The uploader keeps its files when
# it moves, because Streamlit identifies widgets by their arguments, not their position.

datasets = st.session_state.datasets
start = st.expander("Change or add data", expanded=False) if datasets else st.container()
with start:
    if not datasets:
        st.write("")
        st.subheader("Start with a sample or your own file")
    cards_col, upload_col = st.columns([3, 2], gap="large")
    with cards_col:
        for key in ("cancer", "netflix", "uber"):
            info = SAMPLE_SETS[key]
            shape = load_sample(info["files"][0]).shape
            with st.container(border=True, key=f"card-{key}"):
                img_col, text_col = st.columns([1, 5], vertical_alignment="center")
                img_col.image(info["image"], width=56)
                with text_col:
                    st.markdown(f"**{info['name']}**  \n:gray[{shape[0]:,} rows · {shape[1]} columns]")
                    st.caption(info["desc"])
                    in_use = st.session_state.get("last_sample") == key
                    with st.container(horizontal=True):
                        st.button("In use" if in_use else "Use this dataset", key=f"use-{key}",
                                  disabled=in_use, on_click=use_sample, args=(key,))
                        if key == "uber":
                            prices_in_use = st.session_state.get("last_sample") == "uber_prices"
                            st.button("In use with vehicle prices" if prices_in_use
                                      else "Use with vehicle prices (2 files)", key="use-uber_prices",
                                      type="tertiary", disabled=prices_in_use,
                                      on_click=use_sample, args=("uber_prices",),
                                      help="Adds a small made-up price table per vehicle type, to try joins.")
    with upload_col:
        uploads = st.file_uploader("Drop CSV or Excel files here", type=["csv", "xlsx"],
                                   accept_multiple_files=True,
                                   help="You can add several files; the agent can join them.")
        st.caption("Questions that work on most files:  \n"
                   "• Give me a quick overview of this dataset  \n"
                   "• Which *category* has the highest average *value*?  \n"
                   "• Why did *value* change from March to April?  \n"
                   "• Forecast *value* for the next 3 months  \n"
                   "After upload you'll get suggestions based on your columns.")
    st.pills("Or try an example question", list(EXAMPLES), key="example_chip", on_change=run_example)

# Sync uploaded files with the datasets (add new files, drop removed ones).
upload_ids = {f"upload:{f.name}:{f.size}": f for f in uploads or []}
remove_datasets(lambda src: src.startswith("upload:") and src not in upload_ids)
for upload_id, f in upload_ids.items():
    if upload_id in st.session_state.sources.values():
        continue
    try:
        file_io = io.BytesIO(f.getvalue())
        new_df = (pd.read_csv(file_io) if f.name.lower().endswith(".csv")
                  else pd.read_excel(file_io, engine="openpyxl"))
        add_dataset(dataset_name(f.name, st.session_state.datasets), new_df, upload_id)
    except Exception as e:
        st.error(f"Couldn't read {f.name}: {e}. Check that it's a valid CSV or Excel file.")

datasets = st.session_state.datasets
data_changed = any(name not in st.session_state.originals or df is not st.session_state.originals[name]
                   for name, df in datasets.items()) or "agent" in st.session_state.sources.values()

# ---------------------------------------------------------------- sidebar: rules and session

with st.sidebar:
    st.subheader("📌 Saved rules")
    st.caption("Definitions the agent follows when they apply. You can also type `remember: ...` in the chat.")
    with st.form("add_rule", clear_on_submit=True, border=False):
        new_rule = st.text_input("New rule", placeholder="e.g. Revenue = sum of Booking Value",
                                 label_visibility="collapsed")
        if st.form_submit_button("Save rule", width="stretch") and new_rule.strip():
            rule_memory.add(new_rule)
            st.rerun()
    for rule in rule_memory.rules:
        text_col, del_col = st.columns([5, 1], vertical_alignment="center")
        text_col.markdown(md_escape(rule["text"]))
        if del_col.button("", icon=":material/delete:", key=f"del_{rule['id']}", type="tertiary",
                          help="Delete this rule"):
            rule_memory.delete(rule["id"])
            st.rerun()
    if not rule_memory.rules:
        st.caption("No rules yet.")
    elif not rule_memory.path:
        st.caption("Rules last for this session only.")

    st.subheader("🗂️ Session")
    st.caption("Export what you found, or start over.")
    if st.session_state.chat_messages:
        report_html = build_report(st.session_state.chat_messages, {
            "Datasets": ", ".join(f"{n} ({d.shape[0]:,}×{d.shape[1]})" for n, d in datasets.items()),
            "Model": f"{agent.provider} / {agent.model}",
        })
        st.download_button("Export report as HTML", report_html,
                           f"analysis_report_{time.strftime('%Y%m%d_%H%M')}.html", "text/html",
                           width="stretch", help="All questions, answers, charts and agent steps in one file.")
    if data_changed and st.button("Undo data changes", width="stretch",
                                  help="Restore the loaded data and remove tables the agent created."):
        remove_datasets(lambda src: src == "agent")
        st.session_state.datasets.update(st.session_state.originals)
        st.rerun()
    if st.button("Clear conversation", width="stretch"):
        st.session_state.chat_messages = []
        st.session_state.pending_clarification = None
        agent.clear()
        st.rerun()
    with st.expander("What the agent remembers"):
        if not agent.history:
            st.caption("Nothing yet.")
        for msg in agent.history[-4:]:
            role = "You" if msg.type == "human" else "Agent"
            st.caption(f"**{role}:** {md_escape(str(msg.content)[:300])}")

# ---------------------------------------------------------------- helpers


def show_chart(chart, key: str, downloads: bool = True):
    # theme=None keeps the shared chart theme (theme.py) instead of Streamlit's own. Streamlit still fills
    # in background colours the figure leaves unset, so the transparent ones are set explicitly.
    if chart.kind == "plotly":
        chart.obj.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(chart.obj, key=f"plot_{key}", theme=None)
        if downloads:
            st.download_button("Download chart (HTML)", chart.obj.to_html(), f"chart_{key}.html",
                               "text/html", key=f"dl_{key}", type="tertiary")
    elif chart.kind == "matplotlib":
        st.pyplot(chart.obj)
        if downloads:
            buf = io.BytesIO()
            chart.obj.savefig(buf, format="png", bbox_inches="tight")
            st.download_button("Download chart (PNG)", buf.getvalue(), f"chart_{key}.png",
                               "image/png", key=f"dl_{key}", type="tertiary")
    elif chart.kind == "altair":
        st.altair_chart(chart.obj, key=f"plot_{key}", theme=None)


def friendly_error(e: Exception) -> str:
    """Plain-language message for the API errors visitors of the free demo will actually hit."""
    s = str(e)
    if "Request too large" in s or "Error code: 413" in s:
        return ("This request is too big for the free model's per-request limit. Ask a narrower question, "
                "turn off **Plan first** or **Critic check** in the sidebar, or switch model.")
    if any(k in s for k in ("per day", "TPD", "RESOURCE_EXHAUSTED", "quota", "no remaining credits")):
        return ("The free daily limit for this model is used up. Pick another model or provider in the "
                "sidebar (e.g. `openai/gpt-oss-20b` or Gemini), or try again later.")
    if "429" in s or "Rate limit" in s or "503" in s or "overloaded" in s.lower():
        return "The model is busy right now. Wait a minute, then ask again."
    return f"The request failed: {s[:400]}. Rephrase the question or switch model in the sidebar."


def show_step(s):
    icon, label = step_meta(s)
    first = s.output.strip().splitlines()[0] if s.output.strip() else ""
    summary = "" if s.tool in NOTE_STEPS or first.startswith("ERROR") else first[:110]
    st.markdown(f"{icon} **{label}**" + (f" · :gray[{md_escape(summary)}]" if summary else ""))
    if s.tool in NOTE_STEPS:
        st.caption(md_escape(s.output.strip()))
        return
    if s.code and s.tool != "inspect_data":
        st.code(s.code, language="python", wrap_lines=True)
    output = s.output.strip()[:1500]
    st.code(output, language=None, wrap_lines=True, height=220 if output.count("\n") > 10 else "content")


def show_steps(steps, key: str):
    if not steps:
        return
    tools = [s for s in steps if s.tool not in NOTE_STEPS]
    with st.expander(f"How it got this · {len(tools)} step{'s' if len(tools) != 1 else ''}"):
        with st.container(key=f"timeline-{key}"):
            for s in steps:
                show_step(s)


def check_badge(steps):
    critic = [s for s in steps or [] if s.tool == "critic"]
    if critic:
        if critic[0].ok:
            st.badge("Checked", icon="✅", color="green")
        else:
            st.badge("Revised after check", icon="⚠️", color="orange")


def render_assistant(msg, key: str):
    if msg.get("error"):
        st.error(msg["content"], icon="⚠️")
    else:
        st.markdown(highlight_markdown(msg["content"]))
    if msg.get("clarification"):
        st.badge("Needs your answer", icon="❓", color="blue")
    check_badge(msg.get("steps"))
    if msg.get("data_note"):
        st.success(msg["data_note"], icon="✅")
    for j, chart in enumerate(msg.get("charts", [])):
        show_chart(chart, f"{key}_{j}")
    show_steps(msg.get("steps"), key)


# ---------------------------------------------------------------- empty state

if not datasets:
    st.stop()

# ---------------------------------------------------------------- KPI strip

primary_name, primary = next(iter(datasets.items()))
stats_cache = st.session_state.setdefault("kpi_cache", {})
if id(primary) not in stats_cache:  # duplicates on 150k rows takes a moment; compute once per table
    cells = primary.shape[0] * primary.shape[1]
    stats_cache.clear()
    stats_cache[id(primary)] = (primary.isna().sum().sum() / cells * 100 if cells else 0,
                                int(primary.duplicated().sum()))
missing_pct, duplicates = stats_cache[id(primary)]
with st.container(horizontal=True, gap="medium"):
    st.metric("Rows", f"{primary.shape[0]:,}", border=True)
    st.metric("Columns", f"{primary.shape[1]}", border=True)
    st.metric("Missing values", f"{missing_pct:.1f}%", border=True)
    st.metric("Duplicate rows", f"{duplicates:,}", border=True)
if len(datasets) > 1:
    st.caption(f"Figures for `{primary_name}` · {len(datasets)} tables loaded, see the Data tab.")

chat_tab, data_tab, dash_tab = st.tabs(["Chat", "Data", "Dashboard"])

# ---------------------------------------------------------------- data tab

with data_tab:
    def show_table(name, d):
        src = st.session_state.sources.get(name, "")
        origin = ("Sample dataset" if src == "sample" else "Created by the agent" if src == "agent"
                  else f"Uploaded file: {src.split(':')[1]}")
        edited = " · edited in this session" if d is not st.session_state.originals.get(name) else ""
        st.caption(f"`{name}` · {d.shape[0]:,} rows × {d.shape[1]} columns · {origin}{edited}")
        st.dataframe(d.head(100), width="stretch", height=320)
        columns = pd.DataFrame({"Column": d.columns.astype(str), "Type": d.dtypes.astype(str).values,
                                "Missing": d.isna().sum().values, "Unique values": d.nunique().values})
        st.markdown("**Columns**")
        st.dataframe(columns, width="stretch", hide_index=True)

    if len(datasets) == 1:
        show_table(*next(iter(datasets.items())))
    else:
        for tab, (name, d) in zip(st.tabs(list(datasets)), datasets.items()):
            with tab:
                show_table(name, d)

# ---------------------------------------------------------------- dashboard tab

with dash_tab:
    charts = [(i, j, c) for i, m in enumerate(st.session_state.chat_messages)
              for j, c in enumerate(m.get("charts", []))]
    if not charts:
        st.caption("Charts from the chat collect here. Ask for one in the Chat tab, "
                   "e.g. \"Show average risk score by cancer type as a bar chart\".")
    for row in range(0, len(charts), 2):
        for col, (i, j, chart) in zip(st.columns(2, gap="medium"), charts[row:row + 2]):
            with col, st.container(border=True):
                show_chart(chart, f"dash_{i}_{j}", downloads=False)
                if chart.explanation:
                    st.caption(chart.explanation)

# ---------------------------------------------------------------- chat tab

with chat_tab:
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button(AUTO_REPORT_LABEL, type="primary",
                     help="The agent explores the data on its own: key facts, statistics, 2 charts, "
                          "what drives the main change, and a short report."):
            ask(AUTO_REPORT_LABEL)
            st.rerun()
        st.caption("Writes a short report in about 10 steps. Takes 2–4 minutes on the free model tier; "
                   "you can watch each step live.")

    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            if msg["role"] == "assistant":
                render_assistant(msg, str(i))
            else:
                st.markdown(md_escape(msg["content"]))

    # The agent asked a clarifying question: offer its suggested answers as buttons.
    pending = st.session_state.get("pending_clarification")
    if pending and st.session_state.chat_messages and st.session_state.chat_messages[-1].get("clarification"):
        if pending["options"]:
            with st.container(horizontal=True):
                for option in pending["options"]:
                    if st.button(option, key=f"opt_{option}"):
                        ask(option)
                        st.rerun()
        st.checkbox("Remember my answer as a rule", key="remember_answer",
                    help="Saves the question and your answer, so the agent won't need to ask again.")

    if not st.session_state.chat_messages:
        st.pills("Suggested for this data", suggest_questions(datasets), key="suggestion_chip",
                 on_change=run_suggestion)

    prompt = st.chat_input("Type your answer, or pick an option above" if pending
                           else "Ask a question about your data")
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    if prompt and (rule_text := parse_remember(prompt)):
        # "remember: ..." saves a rule; no agent call needed.
        rule_memory.add(rule_text)
        st.session_state.chat_messages += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Saved the rule *{md_escape(rule_text)}*. "
                                             "I'll apply it to related questions."},
        ]
        st.rerun()

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(md_escape(prompt))

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            # Live steps: each tool call is added to this box as soon as it finishes.
            status = st.status("Starting", expanded=True)
            live_steps = []
            ctx = get_script_run_ctx()

            def on_step(step):
                # Tools run in a LangGraph worker thread; attach it to this session so st calls work.
                add_script_run_ctx(threading.current_thread(), ctx)
                live_steps.append(step)
                status.update(label=STEPS.get(step.tool, ("", "", "Working"))[2])
                with status:
                    show_step(step)

            try:
                question = AUTO_REPORT_PROMPT if prompt == AUTO_REPORT_LABEL else prompt
                original = question
                if pending and prompt != AUTO_REPORT_LABEL:
                    # This message answers the agent's clarifying question: continue the original request.
                    original = pending["original"]
                    if st.session_state.get("remember_answer"):
                        rule_memory.add(f"{pending['question']} → {prompt}")
                    question = (f'My answer to your question ("{pending["question"]}"): {prompt}\n'
                                f"Now answer my original request: {original}")
                st.session_state.pending_clarification = None
                rules = [r["text"] for r, _ in rule_memory.relevant(question)]
                # The auto report's instructions already are a plan, so it skips the planning call; it also
                # skips the critic, whose review + fix round would double its run time on the free tier.
                is_report = prompt == AUTO_REPORT_LABEL
                reply = agent.ask(question, dict(datasets), on_step=on_step, critic=use_critic and not is_report,
                                  rules=rules, plan=use_plan and not is_report)
                status.update(label=f"Done in {len(reply.steps)} steps", state="complete", expanded=False)
                msg = {"role": "assistant", "content": reply.answer,
                       "charts": reply.charts, "steps": reply.steps}
                if reply.clarification:
                    msg["content"] = reply.clarification["question"]
                    msg["clarification"] = reply.clarification
                    st.session_state.pending_clarification = {**reply.clarification, "original": original}
                    status.update(label="Needs one detail from you", state="complete")
                if reply.data_changed:
                    notes = []
                    for name, new_df in reply.datasets.items():
                        old_df = datasets.get(name)
                        if old_df is None:
                            add_dataset(name, new_df, "agent")
                            notes.append(f"new table `{name}` with {new_df.shape[0]:,} rows × {new_df.shape[1]} columns")
                        elif new_df is not old_df:
                            datasets[name] = new_df
                            notes.append(f"`{name}` went from {old_df.shape[0]:,}×{old_df.shape[1]} to "
                                         f"{new_df.shape[0]:,}×{new_df.shape[1]}")
                    if notes:
                        msg["data_note"] = "Data updated: " + "; ".join(notes)
            except Exception as e:
                status.update(label="Stopped with an error", state="error", expanded=False)
                msg = {"role": "assistant", "content": friendly_error(e), "steps": live_steps, "error": True}

        # Save first, then rerun so the chat, sidebar and data views all show the new state.
        st.session_state.chat_messages.append(msg)
        st.rerun()
