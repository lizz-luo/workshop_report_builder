import json
import os
import re
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Questionnaire Report Tool", layout="wide")
st.title("Questionnaire Report Tool")
st.caption("Paste question headers and response rows, then use Groq LLM to clarify structure before parsing.")

DEFAULT_CFG = {
    "basic_keywords": ["時間戳記", "日期", "Date", "School region", "School sector", "School name", "學校地區", "學校類別", "學校名稱", "工作坊", "分享會"],
    "ignored_keywords": ["暫時停用"],
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "normalized_questions" not in st.session_state:
    st.session_state.normalized_questions = []
if "normalized_rows" not in st.session_state:
    st.session_state.normalized_rows = []
if "llm_result" not in st.session_state:
    st.session_state.llm_result = ""

st.sidebar.header("Settings")
keep_duplicate_text = st.sidebar.checkbox("Keep duplicate text responses", value=True)
show_raw_preview = st.sidebar.checkbox("Show raw parsed table", value=True)
use_llm = st.sidebar.checkbox("Enable Groq LLM assistant", value=True)
groq_model = st.sidebar.text_input("Groq model", value="llama-3.1-70b-versatile")
groq_api_key = st.sidebar.text_input("Groq API key", type="password")
groq_base_url = st.sidebar.text_input("Groq base URL", value="https://api.groq.com/openai/v1")

col1, col2 = st.columns(2)
with col1:
    q_text = st.text_area("1) Paste question headers, one per line", height=280, placeholder="時間戳記\n分享會日期 Date of the Internal Sharing Session\n...", key="q_text")
with col2:
    r_text = st.text_area("2) Paste response rows, one row per respondent", height=280, placeholder="2026年4月14日 下午03:31:11\n2026年4月14日\n香港 Hong Kong\n...", key="r_text")

st.divider()
st.subheader("LLM clarification chat")
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

chat_msg = st.chat_input("Ask Groq to clarify or normalize the pasted content")
if chat_msg:
    st.session_state.chat_history.append(("user", chat_msg))


def split_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_with_llm(q_text, r_text, chat_history, api_key, model, base_url):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    system = (
        "You are helping normalize pasted questionnaire data for a Python app. "
        "Infer a clean structure from the user's pasted content. "
        "If something is unclear, ask concise clarification questions. "
        "Return ONLY valid JSON with keys: questions, rows, clarification_questions, notes. "
        "questions must be an array of cleaned question strings in order. "
        "rows must be an array of respondent row arrays. "
        "clarification_questions must be an array of short strings. "
        "notes must be a short string."
    )
    messages = [{"role": "system", "content": system}]
    for role, content in chat_history[-8:]:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"Questions:\n{q_text}\n\nResponses:\n{r_text}"})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content


def normalize_llm_output(raw):
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        return None


def parse_plain_inputs(q_text, r_text):
    questions = split_lines(q_text)
    all_lines = split_lines(r_text)
    if not questions or not all_lines:
        return questions, [], pd.DataFrame()
    qn = len(questions)
    nrows = len(all_lines) // qn
    rows = [all_lines[i * qn:(i + 1) * qn] for i in range(nrows)]
    df = pd.DataFrame(rows, columns=questions)
    return questions, rows, df


def is_ignored(q):
    return any(k in q for k in DEFAULT_CFG["ignored_keywords"])


def is_numbered_item(q):
    return bool(re.match(r"^\s*\d+[\.、]\s*", q))


def is_text_item(q):
    keywords = ["文字", "Other comments", "ideas/ strategies", "messages/ ideas", "啟發", "意見", "重點"]
    return any(k in q for k in keywords)


def is_choice_item(q, answers):
    if is_text_item(q):
        return False
    if is_numbered_item(q):
        return True
    if any(k in q for k in ["同意", "滿意", "評價", "share", "workshop", "session", "subject", "section"]):
        return True
    uniq = [a for a in answers if a and a != "/"]
    return len(uniq) >= 2 and len(set(uniq)) <= 8


def classify_questions(questions, rows):
    out = []
    for i, q in enumerate(questions):
        col_answers = [r[i] if i < len(r) else "" for r in rows]
        if is_ignored(q):
            typ = "ignored"
        elif i < 8 and any(k in q for k in DEFAULT_CFG["basic_keywords"]):
            typ = "basic"
        elif any(k in q for k in ["時間戳記", "日期", "School region", "School sector", "School name", "學校地區", "學校類別", "學校名稱"]):
            typ = "basic"
        elif is_text_item(q):
            typ = "text"
        elif is_choice_item(q, col_answers):
            typ = "choice"
        else:
            typ = "text" if i >= 8 else "basic"
        out.append({
            "index": i + 1,
            "question": q,
            "type": typ,
            "sample_values": ", ".join(sorted(set([v for v in col_answers if v]))[:5]),
        })
    return pd.DataFrame(out)


if use_llm and q_text and r_text:
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Groq is normalizing your input..."):
            raw = parse_with_llm(q_text, r_text, st.session_state.chat_history, groq_api_key, groq_model, groq_base_url)
            st.session_state.llm_result = raw
            parsed = normalize_llm_output(raw)
        st.text_area("Groq raw output", raw, height=220)
        if parsed:
            questions = parsed.get("questions") or split_lines(q_text)
            rows = parsed.get("rows") or []
            if rows:
                df = pd.DataFrame(rows, columns=questions[:len(rows[0])])
                st.session_state.normalized_questions = questions
                st.session_state.normalized_rows = rows
                classify_df = classify_questions(questions, rows)
                basic_qs = classify_df[classify_df["type"] == "basic"]
                choice_qs = classify_df[classify_df["type"] == "choice"]
                text_qs = classify_df[classify_df["type"] == "text"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Respondents", len(df))
                c2.metric("Basic fields", len(basic_qs))
                c3.metric("Choice items", len(choice_qs))
                c4.metric("Text items", len(text_qs))
                st.subheader("Auto classification")
                st.dataframe(classify_df, use_container_width=True, hide_index=True)
                if show_raw_preview:
                    st.subheader("Raw parsed table")
                    st.dataframe(df, use_container_width=True)
                st.subheader("Choice item summary")
                for _, row in choice_qs.iterrows():
                    q = row["question"]
                    s = df[q].replace("", pd.NA).dropna()
                    counts = s.value_counts(dropna=False)
                    total = counts.sum()
                    if total:
                        out_df = pd.DataFrame({"Response": counts.index, "Count": counts.values, "Percent": (counts.values / total * 100).round(1)})
                    else:
                        out_df = pd.DataFrame(columns=["Response", "Count", "Percent"])
                    st.markdown(f"**{q}**")
                    st.dataframe(out_df, use_container_width=True, hide_index=True)
                st.subheader("Text item responses")
                for _, row in text_qs.iterrows():
                    q = row["question"]
                    vals = [v for v in df[q].tolist() if v and v != "/"]
                    if not keep_duplicate_text:
                        vals = sorted(set(vals))
                    else:
                        c = Counter(vals)
                        vals = [f"{x} [count: {c[x]}]" if c[x] > 1 else x for x in sorted(c.keys())]
                    st.markdown(f"**{q}**")
                    st.write("\n".join([f"- {x}" for x in vals]) if vals else "(no text responses)")
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Download parsed CSV", csv, file_name="parsed_responses.csv", mime="text/csv")
            else:
                st.info("Groq returned questions but no row data was normalized.")
        else:
            st.warning("Groq output was not valid JSON. Please refine the prompt or ask a clarification question.")
        if chat_msg:
            st.session_state.chat_history.append(("assistant", raw))

elif q_text and r_text and not use_llm:
    questions, rows, df = parse_plain_inputs(q_text, r_text)
    if not df.empty:
        classify_df = classify_questions(questions, rows)
        basic_qs = classify_df[classify_df["type"] == "basic"]
        choice_qs = classify_df[classify_df["type"] == "choice"]
        text_qs = classify_df[classify_df["type"] == "text"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Respondents", len(df))
        c2.metric("Basic fields", len(basic_qs))
        c3.metric("Choice items", len(choice_qs))
        c4.metric("Text items", len(text_qs))
        st.subheader("Auto classification")
        st.dataframe(classify_df, use_container_width=True, hide_index=True)
        if show_raw_preview:
            st.subheader("Raw parsed table")
            st.dataframe(df, use_container_width=True)
        st.subheader("Choice item summary")
        for _, row in choice_qs.iterrows():
            q = row["question"]
            s = df[q].replace("", pd.NA).dropna()
            counts = s.value_counts(dropna=False)
            total = counts.sum()
            if total:
                out_df = pd.DataFrame({"Response": counts.index, "Count": counts.values, "Percent": (counts.values / total * 100).round(1)})
            else:
                out_df = pd.DataFrame(columns=["Response", "Count", "Percent"])
            st.markdown(f"**{q}**")
            st.dataframe(out_df, use_container_width=True, hide_index=True)
        st.subheader("Text item responses")
        for _, row in text_qs.iterrows():
            q = row["question"]
            vals = [v for v in df[q].tolist() if v and v != "/"]
            if not keep_duplicate_text:
                vals = sorted(set(vals))
            else:
                c = Counter(vals)
                vals = [f"{x} [count: {c[x]}]" if c[x] > 1 else x for x in sorted(c.keys())]
            st.markdown(f"**{q}**")
            st.write("\n".join([f"- {x}" for x in vals]) if vals else "(no text responses)")
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download parsed CSV", csv, file_name="parsed_responses.csv", mime="text/csv")
else:
    st.info("Paste question headers and response rows to begin.")