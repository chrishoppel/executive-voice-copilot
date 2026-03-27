import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.audio import synthesize_speech, transcribe_audio
from src.coaching import build_coach_response, detect_fillers, save_session

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

load_dotenv()

api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Executive Communication Coach",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        padding-top: 0.8rem;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 1.25rem 1.35rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
    }
    .section-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .soft-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        color: #0f172a;
        margin-bottom: 0.8rem;
    }
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        color: #0f172a;
        margin-bottom: 0.8rem;
    }
    .muted {
        color: #64748b;
        font-size: 0.92rem;
    }
    .footer-note {
        color: #94a3b8;
        font-size: 0.84rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to Streamlit secrets or your local .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

SITUATIONS = [
    "Executive update",
    "Interview answer",
    "Recommendation to leadership",
    "Difficult stakeholder conversation",
    "Board / CFO challenge",
]

SAMPLE_PROMPTS = {
    "Executive update": "I need to explain to leadership that reporting inconsistency is slowing decisions because the team keeps reconciling numbers instead of acting on the data",
    "Interview answer": "I need to explain how I lead analytics teams in a way that improves decision quality, not just reporting output",
    "Recommendation to leadership": "I need to explain to the CFO that we should not cut analytics resources right now because reducing capacity will slow reporting, weaken decision support, and hurt spend optimization",
    "Difficult stakeholder conversation": "I need to tell a cross-functional leader that the current request is not scoped well enough to deliver a reliable answer quickly",
    "Board / CFO challenge": "I need to explain why fragmented customer data is limiting personalization and reducing marketing efficiency",
}

STATE_DEFAULTS = {
    "manual_text": "",
    "transcript_text": "",
    "coach": None,
    "challenge_result": None,
    "challenge_input": "",
    "filler_counts": {},
    "playback_path": None,
    "session_path": None,
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_session() -> None:
    for key, value in STATE_DEFAULTS.items():
        st.session_state[key] = value


def build_why_stronger_points(coach, transcript_text: str) -> list[str]:
    points = []
    if transcript_text:
        points.append("It leads with the main point faster")
    if coach.why_it_matters:
        points.append("It makes the business consequence clearer")
    if coach.recommendation:
        points.append("It gives a stronger point of view")
    if coach.stronger_closing_line:
        points.append("It ends more confidently")
    return points[:3]


def run_coaching(
    *,
    audio_value,
    manual_text: str,
    situation: str,
    audience: str,
    tone: str,
    response_length: str,
    mode: str,
    tts_voice: str,
    save_history: bool,
) -> None:
    transcript_text = ""

    if audio_value is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = OUTPUT_DIR / f"input_{timestamp}.wav"
        input_path.write_bytes(audio_value.getvalue())
        with st.spinner("Transcribing audio..."):
            transcript_text = transcribe_audio(client, input_path)
    elif manual_text.strip():
        transcript_text = manual_text.strip()
    else:
        st.warning("Record audio or paste text first.")
        return

    if len(transcript_text.split()) < 6:
        st.warning("Add a little more context so the coaching can produce a specific response.")
        return

    actual_mode = mode
    if situation == "Board / CFO challenge" and mode == "Practice":
        actual_mode = "Challenge"

    with st.spinner("Sharpening your answer..."):
        coach = build_coach_response(
            client=client,
            transcript=transcript_text,
            audience=audience,
            tone=tone,
            response_length=response_length,
            mode=actual_mode,
        )

    filler_counts = detect_fillers(transcript_text)
    playback_path = OUTPUT_DIR / "executive_playback.mp3"

    with st.spinner("Generating audio playback..."):
        synthesize_speech(
            client,
            coach.polished_spoken_version,
            playback_path,
            voice=tts_voice,
        )

    session_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "situation": situation,
            "audience": audience,
            "tone": tone,
            "response_length": response_length,
            "mode": actual_mode,
            "voice": tts_voice,
        },
        "transcript": transcript_text,
        "filler_counts": filler_counts,
        "coach_response": coach.model_dump(),
    }

    session_path = None
    if save_history:
        session_path = save_session(session_payload, DATA_DIR)

    st.session_state.transcript_text = transcript_text
    st.session_state.coach = coach
    st.session_state.challenge_result = None
    st.session_state.challenge_input = ""
    st.session_state.filler_counts = filler_counts
    st.session_state.playback_path = str(playback_path)
    st.session_state.session_path = session_path.name if session_path else None
    st.session_state.manual_text = manual_text


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Executive Communication Coach</div>
        <div class="hero-subtitle">Practice turning rough thoughts into clear, confident, leadership-ready communication</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Setup")
    situation = st.selectbox("Situation", SITUATIONS)
    audience = st.selectbox("Audience", ["CEO", "CMO", "CFO", "Board", "Peer Leader", "Recruiter"])
    tone = st.selectbox("Tone", ["Decisive", "Calm", "Strategic", "Persuasive"])
    response_length = st.selectbox("Length", ["20 seconds", "45 seconds", "90 seconds", "2 minutes"])
    mode = st.selectbox("Mode", ["Practice", "Reframe", "Challenge", "Polish"])
    tts_voice = st.selectbox("Voice", ["cedar", "marin", "alloy", "ash", "echo", "sage"])
    save_history = st.checkbox("Save session history", value=True)
    st.divider()
    if st.button("Clear session", use_container_width=True):
        clear_session()
        st.rerun()

top_left, top_right = st.columns([3, 2])

with top_left:
    st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="soft-card">
            Choose a situation, then type or record the way you would actually say it.
            The goal is not perfection on the first try. The goal is improvement.
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    sample_choice = st.selectbox("Sample prompt", ["None"] + SITUATIONS)
    if sample_choice != "None":
        if st.button("Use sample", use_container_width=True):
            st.session_state.manual_text = SAMPLE_PROMPTS[sample_choice]
            st.rerun()

input_col, helper_col = st.columns([3, 2])

with input_col:
    st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
    st.subheader("Your draft")

    audio_value = st.audio_input("Record your answer")

    manual_text = st.text_area(
        "Or paste your rough answer",
        height=180,
        value=st.session_state.manual_text,
        placeholder="Type the way you would naturally say it, even if it feels rough or incomplete...",
    )

    btn_cols = st.columns([2, 1])
    with btn_cols[0]:
        sharpen = st.button("Sharpen my answer", type="primary", use_container_width=True)
    with btn_cols[1]:
        if st.button("Clear input", use_container_width=True):
            st.session_state.manual_text = ""
            st.rerun()

with helper_col:
    st.markdown('<div class="section-label">What you will get</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="soft-card">
            <div>• A sharper version of your answer</div>
            <div>• A quick explanation of why it improved</div>
            <div>• One realistic challenge question</div>
            <div>• A chance to practice again immediately</div>
            <div>• Audio playback for delivery practice</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if sharpen:
    run_coaching(
        audio_value=audio_value,
        manual_text=manual_text,
        situation=situation,
        audience=audience,
        tone=tone,
        response_length=response_length,
        mode=mode,
        tts_voice=tts_voice,
        save_history=save_history,
    )

coach = st.session_state.coach
transcript_text = st.session_state.transcript_text
filler_counts = st.session_state.filler_counts
playback_path = st.session_state.playback_path
session_name = st.session_state.session_path

if coach:
    why_stronger = build_why_stronger_points(coach, transcript_text)

    st.divider()
    st.markdown('<div class="section-label">Step 3</div>', unsafe_allow_html=True)
    st.subheader("Your sharper answer")

    result_left, result_right = st.columns([3, 2])

    with result_left:
        st.audio(playback_path, format="audio/mp3")

        st.markdown('<div class="section-label">Sharpened version</div>', unsafe_allow_html=True)
        st.code(coach.polished_spoken_version, language=None)

        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button(
                "Download answer",
                data=coach.polished_spoken_version,
                file_name="sharpened_answer.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl_cols[1]:
            st.download_button(
                "Download session",
                data=json.dumps(
                    {
                        "situation": situation,
                        "transcript": transcript_text,
                        "headline": coach.executive_headline,
                        "business_impact": coach.why_it_matters,
                        "recommendation": coach.recommendation,
                        "spoken_version": coach.polished_spoken_version,
                        "challenge": coach.tough_question,
                    },
                    indent=2,
                ),
                file_name="communication_session.json",
                mime="application/json",
                use_container_width=True,
            )

        if session_name:
            st.caption(f"Saved session: {session_name}")

    with result_right:
        st.markdown('<div class="section-label">Why it improved</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        for point in why_stronger:
            st.write(f"- {point}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Coach’s challenge</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-card">{coach.tough_question}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">One thing to improve next</div>', unsafe_allow_html=True)
        next_improvement = coach.coaching_feedback[0] if coach.coaching_feedback else "Make the main point earlier and keep the recommendation direct."
        st.markdown(f'<div class="result-card">{next_improvement}</div>', unsafe_allow_html=True)

    compare_tab, practice_tab, details_tab = st.tabs(["Compare", "Practice again", "Details"])

    with compare_tab:
        comp_left, comp_right = st.columns(2)

        with comp_left:
            st.markdown('<div class="section-label">Original</div>', unsafe_allow_html=True)
            st.write(transcript_text)

        with comp_right:
            st.markdown('<div class="section-label">Sharpened</div>', unsafe_allow_html=True)
            st.write(coach.polished_spoken_version)

    with practice_tab:
        st.markdown('<div class="section-label">Answer the challenge</div>', unsafe_allow_html=True)
        st.write("Respond to the challenge question in your own words, then sharpen that answer too.")

        challenge_input = st.text_area(
            "Your response",
            height=150,
            value=st.session_state.challenge_input,
            placeholder="Type how you would answer the challenge question...",
        )

        practice_btn_cols = st.columns([2, 1])
        with practice_btn_cols[0]:
            practice_again = st.button("Sharpen my challenge response", use_container_width=True)
        with practice_btn_cols[1]:
            if st.button("Reset practice", use_container_width=True):
                st.session_state.challenge_input = ""
                st.session_state.challenge_result = None
                st.rerun()

        if practice_again:
            st.session_state.challenge_input = challenge_input
            if len(challenge_input.split()) < 6:
                st.warning("Add a little more detail so the challenge response can be sharpened.")
            else:
                with st.spinner("Sharpening challenge response..."):
                    st.session_state.challenge_result = build_coach_response(
                        client=client,
                        transcript=challenge_input,
                        audience=audience,
                        tone=tone,
                        response_length=response_length,
                        mode="Polish",
                    )

        if st.session_state.challenge_result:
            retry_result = st.session_state.challenge_result

            pr_left, pr_right = st.columns(2)
            with pr_left:
                st.markdown('<div class="section-label">Your challenge response</div>', unsafe_allow_html=True)
                st.write(st.session_state.challenge_input)

            with pr_right:
                st.markdown('<div class="section-label">Sharpened challenge response</div>', unsafe_allow_html=True)
                st.write(retry_result.polished_spoken_version)

    with details_tab:
        det_left, det_right = st.columns(2)

        with det_left:
            st.subheader("Headline")
            st.write(coach.executive_headline)

            st.subheader("Business impact")
            st.write(coach.why_it_matters)

            st.subheader("Recommendation")
            st.write(coach.recommendation)

        with det_right:
            st.subheader("Supporting points")
            for point in coach.support_points:
                st.write(f"- {point}")

            st.subheader("Filler phrases detected")
            if filler_counts:
                for phrase, count in filler_counts.items():
                    st.write(f"- {phrase}: {count}")
            else:
                st.write("No common filler phrases detected.")

        st.divider()
        score_cols = st.columns(5)
        score_cols[0].metric("Clarity", coach.scores.clarity)
        score_cols[1].metric("Concision", coach.scores.concision)
        score_cols[2].metric("Presence", coach.scores.executive_presence)
        score_cols[3].metric("Business", coach.scores.business_focus)
        score_cols[4].metric("Action", coach.scores.actionability)

else:
    st.divider()
    st.markdown('<div class="muted">Start with a rough answer above, then click <strong>Sharpen my answer</strong>.</div>', unsafe_allow_html=True)

st.divider()
st.markdown(
    '<div class="footer-note">Executive Communication Coach • Practice clear, confident communication over time</div>',
    unsafe_allow_html=True,
)