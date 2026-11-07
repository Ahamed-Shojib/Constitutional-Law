"""Module 7 - Web UI.  Run from the project root:  streamlit run ui/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

from app.pipeline import LegalRAG

st.set_page_config(page_title="Constitution of BD", page_icon="⚖️", layout="centered")


@st.cache_resource
def get_rag():
    return LegalRAG()


st.title("⚖️ Constitution of BD")
st.caption("Describe what happened (বাংলা বা English). Get the relevant constitutional rights, laws and what to do next.")

examples = ["Someone snatched my phone on the road", "পুলিশ আমার ভাইকে গ্রেপ্তার করেছে কিন্তু কারণ জানায়নি",
            "My employer has not paid my salary for 3 months"]
choice = st.selectbox("Try an example", [""] + examples)
query = st.text_area("Your problem", value=choice, height=120, placeholder="What happened? When? Where? Who was involved?")

if st.button("Find applicable law", type="primary") and query.strip():
    a = get_rag().ask(query)
    for w in a.warnings[:1] if a.emergency else []:
        st.error(w)
    st.subheader(a.category_label)
    st.caption(f"Classifier confidence: {a.confidence:.0%}")
    st.write(a.explanation)

    if a.constitutional_articles:
        st.markdown("### Constitutional rights involved")
        for c in a.constitutional_articles:
            with st.expander(f"{c.label} - {c.title}"):
                st.write(c.snippet)
    if a.applicable_laws:
        st.markdown("### Applicable laws")
        for c in a.applicable_laws:
            with st.expander(f"{c.label} - {c.title}"):
                st.write(c.snippet)
                if not c.official_text:
                    st.caption("Summary only - verify official text on bdlaws.minlaw.gov.bd")
    st.markdown("### What to do now")
    for i, s in enumerate(a.steps, 1):
        st.markdown(f"**{i}.** {s}")
    st.markdown("### Helplines")
    for h in a.helplines:
        st.markdown(f"- **{h['number']}** - {h['name']}")
    for w in (a.warnings[1:] if a.emergency else a.warnings):
        st.warning(w)
    st.info(a.disclaimer)
