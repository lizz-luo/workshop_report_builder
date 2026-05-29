import streamlit as st
import pandas as pd
import re
from collections import Counter
from io import StringIO

st.set_page_config(page_title="Questionnaire Report Tool", layout="wide")
st.title("Questionnaire Report Tool")
st.caption("Paste question headers and response rows, then preview, classify, and generate report-ready outputs.")

DEFAULT_CFG = {
    "basic_keywords": ["時間戳記", "日期", "Date", "School region", "School sector", "School name", "學校地區", "學校類別", "學校名稱", "工作坊", "分享會"],
    "ignored_keywords": ["暫時停用"],
    "scale_options": ["非常同意", "同意", "不同意", "非常不同意", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree"],
}

st.sidebar.header("Settings")
keep_duplicate_text = st.sidebar.checkbox("Keep duplicate text responses", value=True)
show_raw_preview = st.sidebar.checkbox("Show raw parsed table", value=True)

q_text = st.text_area("1) Paste question headers, one per line", height=300, placeholder="時間戳記\n分享會日期 Date of the Internal Sharing Session\n...", key="q_text")
r_text = st.text_area("2) Paste response rows, one row per respondent", height=300, placeholder="2026年4月14日 下午03:31:11\n2026年4月14日\n香港 Hong Kong\n...", key="r_text")


def split_lines(text):
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s:
            lines.append(s)
    return lines


def is_ignored(q):
    return any(k in q for k in DEFAULT_CFG["ignored_keywords"])


def is_numbered_item(q):
    return bool(re.match(r'^\s*\d+[\.、]\s*', q))


def is_text_item(q):
    return any(k in q for k in ["文字", "Other comments", "ideas/ strategies", "messages/ ideas", "啟發", "意見", "重點"])


def is_choice_item(q, answers):
    if is_text_item(q):
        return False
    if is_numbered_item(q):
        return True
    if any(k in q for k in ["同意", "滿意", "評價", "share", "workshop", "session", "subject", "section"]):
        return True
    unique_vals = [a for a in answers if a and a != "/"]
    if len(unique_vals) >= 2 and len(set(unique_vals)) <= 8:
        return True
    return False


def classify_questions(questions, responses):
    rows = []
    for i, q in enumerate(questions):
        col_answers = [r[i] if i < len(r) else "" for r in responses]
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
            typ = "basic" if i < 10 else "text"
        rows.append({"index": i + 1, "question": q, "type": typ, "sample_values": ", ".join(sorted(set([v for v in col_answers if v]))[:5])})
    return pd.DataFrame(rows)


def parse_inputs(q_text, r_text):
    questions = split_lines(q_text)
    all_lines = split_lines(r_text)
    if not questions or not all_lines:
        return questions, [], pd.DataFrame(), []

    qn = len(questions)
    if len(all_lines) % qn != 0:
        st.warning(f"Response lines ({len(all_lines)}) is not a multiple of question count ({qn}). I will use the maximum complete rows only.")
    nrows = len(all_lines) // qn
    rows = [all_lines[i*qn:(i+1)*qn] for i in range(nrows)]
    df = pd.DataFrame(rows, columns=questions)
    return questions, rows, df, []


if q_text and r_text:
    questions, rows, df, _ = parse_inputs(q_text, r_text)
    if questions and not df.empty:
        classify_df = classify_questions(questions, rows)
        basic_qs = classify_df[classify_df["type"] == "basic"]
        choice_qs = classify_df[classify_df["type"] == "choice"]
        text_qs = classify_df[classify_df["type"] == "text"]
        ignored_qs = classify_df[classify_df["type"] == "ignored"]

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

        st.subheader("Basic info preview")
        if not basic_qs.empty:
            basic_preview = {}
            for _, row in basic_qs.iterrows():
                q = row["question"]
                vals = [v for v in df[q].tolist() if v and v != "/"]
                basic_preview[q] = vals[:5]
            st.json(basic_preview)

        st.subheader("Choice item summary")
        if not choice_qs.empty:
            for _, row in choice_qs.iterrows():
                q = row["question"]
                s = df[q].replace("", pd.NA).dropna()
                counts = s.value_counts(dropna=False)
                total = counts.sum()
                out = pd.DataFrame({"Response": counts.index, "Count": counts.values, "Percent": (counts.values / total * 100).round(1)})
                st.markdown(f"**{q}**")
                st.dataframe(out, use_container_width=True, hide_index=True)

        st.subheader("Text item responses")
        if not text_qs.empty:
            for _, row in text_qs.iterrows():
                q = row["question"]
                s = [v for v in df[q].tolist() if v and v != "/"]
                if not keep_duplicate_text:
                    s = sorted(set(s))
                else:
                    c = Counter(s)
                    sorted_vals = sorted(c.keys())
                    s = [f"{x} [count: {c[x]}]" if c[x] > 1 else x for x in sorted_vals]
                st.markdown(f"**{q}**")
                st.write("\n".join([f"- {x}" for x in s]) if s else "(no text responses)")

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download parsed CSV", csv, file_name="parsed_responses.csv", mime="text/csv")
else:
    st.info("Paste question headers and response rows to begin.")
