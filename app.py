"""
Streamlit front-end for the multi-agent research pipeline.

Runs the same 4-stage pipeline as pipeline.py (search agent -> reader agent
-> writer chain -> critic chain) but streams each stage into its own card as
it completes, instead of blocking on one long spinner.
"""

import re
import time
import traceback

import streamlit as st

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Desk",
    page_icon="\U0001F4C1",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------
# Theme: Honk display type + Uiverse sticker-button system
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Honk:MORF@15&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink: #ffffff;          /* White text */
    --paper: #000000;        /* Pure black background */
    --card: #121212;         /* Very dark gray for card surfaces */
    --yellow: #fbca1f;
    --moss: #4a7c59;
    --brick: #e55e4e;        /* Slightly brightened red for dark accessibility */
    --border-w: 3px;
    --shadow-off: 0.22em;
}

/* ---------- base canvas ---------- */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--paper) !important;
    color: var(--ink) !important;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1000px; }

* { font-family: 'Inter', sans-serif; }
p, li, span, label, div { color: var(--ink) !important; }

/* ---------- masthead ---------- */
.masthead {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    border-bottom: var(--border-w) solid var(--ink);
    padding-bottom: 0.6rem;
    margin-bottom: 1.6rem;
}
.masthead-title {
    font-family: 'Honk', 'Space Grotesk', sans-serif;
    font-size: 4.2rem;
    line-height: 0.95;
    margin: 0;
    color: var(--ink);
    letter-spacing: 0.01em;
}
.masthead-sub {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    background: var(--black);
    color: var(--paper);
    padding: 0.3em 0.7em;
    border-radius: 0.3em;
    white-space: nowrap;
}

/* ---------- sticker card ---------- */
.stickercard {
    background: var(--card);
    border: var(--border-w) solid var(--ink);
    border-radius: 0.6em;
    box-shadow: var(--shadow-off) var(--shadow-off) 0 var(--ink);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.8rem;
    position: relative;
}
.stickercard.dim {
    opacity: 0.3;
    box-shadow: none;
    border-style: dashed;
}
.tab {
    position: absolute;
    top: -1.05em;
    left: 1.2em;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    background: var(--yellow);
    border: var(--border-w) solid var(--ink);
    border-radius: 0.35em;
    padding: 0.25em 0.65em;
    box-shadow: 0.1em 0.1em 0 var(--ink);
    color: #000000 !important; /* Keep badge text readable */
}
.tab.moss { background: var(--moss); color: #ffffff !important; }
.tab.brick { background: var(--brick); color: #ffffff !important; }
.tab.dim-tab { background: #262626; color: #737373; box-shadow: none; border-style: dashed; }

.stage-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    margin: 0.5rem 0 0.7rem 0;
}
.stage-body {
    font-size: 0.94rem;
    line-height: 1.55;
    white-space: pre-wrap;
}
.stage-body a { color: var(--brick); }
.stage-body em { color: #888888 !important; }

.score-badge {
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    background: var(--yellow);
    color: #000000 !important;
    border: var(--border-w) solid var(--ink);
    border-radius: 0.4em;
    padding: 0.15em 0.55em;
    box-shadow: 0.12em 0.12em 0 var(--ink);
    margin-bottom: 0.8rem;
    transform: rotate(-2deg);
}

/* ---------- inputs ---------- */
.stTextInput input {
    border: var(--border-w) solid var(--ink) !important;
    border-radius: 0.4em !important;
    padding: 0.7em 0.9em !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1.05rem !important;
    background: var(--card) !important;
    color: var(--ink) !important;
    box-shadow: 0.12em 0.12em 0 var(--ink) !important;
}
.stTextInput input:focus {
    outline: none !important;
    box-shadow: 0.18em 0.18em 0 var(--ink) !important;
}

st.markdown(
    "<p style='margin-top:-0.6rem; margin-bottom:1.4rem; font-family:\"Space Grotesk\", sans-serif; "
    "font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:#aaa; font-size:0.9rem;'>"
    "Search agent finds sources &rarr; Reader agent scrapes the best one "
    "&rarr; Writer drafts the report &rarr; Critic scores it.</p>",
    unsafe_allow_html=True,
)
div[data-testid="stButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translate(-0.05em, -0.05em);
    box-shadow: 0.15em 0.15em 0 var(--ink) !important;
    background: var(--yellow) !important;
    border: var(--border-w) solid var(--ink) !important;
}
div[data-testid="stButton"] > button:active,
div[data-testid="stFormSubmitButton"] > button:active {
    transform: translate(0.05em, 0.05em);
    box-shadow: 0.05em 0.05em 0 var(--ink) !important;
}

/* secondary (download) buttons */
div[data-testid="stDownloadButton"] > button {
    background: var(--card) !important;
    color: var(--ink) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    border: var(--border-w) solid var(--ink) !important;
    border-radius: 0.4em !important;
    box-shadow: 0.1em 0.1em 0 var(--ink) !important;
    transition: transform 0.06s ease, box-shadow 0.06s ease;
}
div[data-testid="stDownloadButton"] > button:hover {
    transform: translate(-0.05em, -0.05em);
    box-shadow: 0.15em 0.15em 0 var(--ink) !important;
}

/* ---------- misc ---------- */
[data-testid="stStatusWidget"] { display: none; }
.stAlert { border: var(--border-w) solid var(--ink) !important; border-radius: 0.4em !important; background: var(--card) !important; }
hr { border-top: var(--border-w) dashed var(--ink); }
footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def render_card(number: str, label: str, color_class: str, heading: str, body: str, dim: bool = False):
    """Render one pipeline-stage sticker card."""
    card_cls = "stickercard dim" if dim else "stickercard"
    tab_cls = f"tab dim-tab" if dim else f"tab {color_class}"
    body_html = body if body else "<em>Waiting for its turn...</em>"
    st.markdown(
        f"""
        <div class="{card_cls}">
            <div class="{tab_cls}">{number} · {label}</div>
            <div class="stage-heading">{heading}</div>
            <div class="stage-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def extract_score(feedback_text: str) -> str:
    match = re.search(r"score\s*:\s*(\d{1,2}\s*/\s*10)", feedback_text, re.IGNORECASE)
    return match.group(1).replace(" ", "") if match else ""


def escape_for_html(text: str) -> str:
    """Minimal escaping so raw report text doesn't break the card's HTML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "search_results": "",
    "scraped_content": "",
    "report": "",
    "feedback": "",
    "running": False,
    "topic_run": "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# --------------------------------------------------------------------------
# Masthead
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <div class="masthead-title">Research Desk</div>
        <div class="masthead-sub">4-Agent Pipeline</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='margin-top:-0.6rem; margin-bottom:1.4rem; font-family:Space Grotesk,sans-serif; "
    "font-weight:500; color:#555;'>Search agent finds sources &rarr; Reader agent scrapes the best one "
    "&rarr; Writer drafts the report &rarr; Critic scores it.</p>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
input_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
with input_col:
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. the state of solid-state batteries in 2026",
        label_visibility="collapsed",
        disabled=st.session_state.running,
    )
with button_col:
    run_clicked = st.button(
        "Run \U0001F50D",
        use_container_width=True,
        disabled=st.session_state.running,
    )

st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------
if run_clicked:
    if not topic or not topic.strip():
        st.warning("Type a topic before running the pipeline.")
    else:
        st.session_state.running = True
        st.session_state.topic_run = topic.strip()
        st.session_state.search_results = ""
        st.session_state.scraped_content = ""
        st.session_state.report = ""
        st.session_state.feedback = ""
        st.rerun()

if st.session_state.running:
    topic = st.session_state.topic_run
    placeholder = st.empty()

    def draw(stage: int):
        """Redraw all four cards; stages beyond `stage` render dim/pending."""
        with placeholder.container():
            render_card(
                "01", "Search Agent", "moss",
                "Scouting sources",
                escape_for_html(st.session_state.search_results)[:1200] if stage >= 1 else "",
                dim=(stage < 1),
            )
            render_card(
                "02", "Reader Agent", "moss",
                "Scraping the top result",
                escape_for_html(st.session_state.scraped_content)[:1200] if stage >= 2 else "",
                dim=(stage < 2),
            )
            render_card(
                "03", "Writer", "", "Drafting the report",
                escape_for_html(st.session_state.report) if stage >= 3 else "",
                dim=(stage < 3),
            )
            score = extract_score(st.session_state.feedback) if stage >= 4 else ""
            score_html = f'<div class="score-badge">{score}</div>' if score else ""
            render_card(
                "04", "Critic", "brick", "Verdict",
                (score_html + escape_for_html(st.session_state.feedback)) if stage >= 4 else "",
                dim=(stage < 4),
            )

    draw(0)

    try:
        # ---- Stage 1: search agent ----
        search_agent = build_search_agent()
        search_result = search_agent.invoke(
            {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
        )
        st.session_state.search_results = search_result["messages"][-1].content
        draw(1)

        # ---- Stage 2: reader agent ----
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{st.session_state.search_results[:800]}",
                    )
                ]
            }
        )
        st.session_state.scraped_content = reader_result["messages"][-1].content
        draw(2)

        # ---- Stage 3: writer chain ----
        research_combined = (
            f"SEARCH RESULTS:\n{st.session_state.search_results}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{st.session_state.scraped_content}"
        )
        st.session_state.report = writer_chain.invoke({"topic": topic, "research": research_combined})
        draw(3)

        # ---- Stage 4: critic chain ----
        st.session_state.feedback = critic_chain.invoke({"report": st.session_state.report})
        draw(4)

    except Exception as exc:
        st.error(f"Pipeline stopped: {exc}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())

    st.session_state.running = False
    time.sleep(0.05)
    st.rerun()

else:
    # idle state: show whatever's in session (finished run, or empty pending cards)
    stage_reached = 0
    if st.session_state.feedback:
        stage_reached = 4
    elif st.session_state.report:
        stage_reached = 3
    elif st.session_state.scraped_content:
        stage_reached = 2
    elif st.session_state.search_results:
        stage_reached = 1

    render_card(
        "01", "Search Agent", "moss", "Scouting sources",
        escape_for_html(st.session_state.search_results)[:1200] if stage_reached >= 1 else "",
        dim=(stage_reached < 1),
    )
    render_card(
        "02", "Reader Agent", "moss", "Scraping the top result",
        escape_for_html(st.session_state.scraped_content)[:1200] if stage_reached >= 2 else "",
        dim=(stage_reached < 2),
    )
    render_card(
        "03", "Writer", "", "Drafting the report",
        escape_for_html(st.session_state.report) if stage_reached >= 3 else "",
        dim=(stage_reached < 3),
    )
    score = extract_score(st.session_state.feedback) if stage_reached >= 4 else ""
    score_html = f'<div class="score-badge">{score}</div>' if score else ""
    render_card(
        "04", "Critic", "brick", "Verdict",
        (score_html + escape_for_html(st.session_state.feedback)) if stage_reached >= 4 else "",
        dim=(stage_reached < 4),
    )

    if st.session_state.report:
        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download report \U0001F4C4",
            data=st.session_state.report,
            file_name=f"{st.session_state.topic_run.replace(' ', '_')[:40] or 'report'}.md",
            mime="text/markdown",
        )