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
    page_title="Executive Voice Copilot",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        padding-top: 1.2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 1.4rem 1.5rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
        margin-bottom: 0.8rem;
    }
    .hero-note {
        font-size: 0.92rem;
        opacity: 0.85;
    }
    .section-card {
        background: #f8fafc;
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .small-label {
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 0.2rem;
    }
    .sample-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.8rem;
        min-height: 220px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to Streamlit secrets or your local .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

DEFAULT_SAMPLES = {
    "CFO Resource Case": (
        "I need to explain to the CFO that we should not cut analytics resources right now "
        "because the team is already stretched and reducing capacity will slow reporting, "
        "weaken decision support, and hurt our ability to optimize spend"
    ),
    "Leadership Reporting Issue": (
        "I need to explain to leadership that reporting inconsistency is slowing decisions "
        "because the team keeps reconciling numbers instead of acting on the data"
    ),
    "Customer Data Fragmentation": (
        "I need to explain why customer data fragmentation is blocking personalization and "
        "reducing campaign efficiency across the business"
    ),
}

for key, value in {
    "transcript_text": "",
    "coach": None,
    "filler_counts": {},
    "playback_path": None,
    "session_path": None,
    "manual_text": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_session() -> None:
    st.session_state.transcript_text = ""
    st.session_state.coach = None
    st.session_state.filler_counts = {}
    st.session_state.playback_path = None
    st.session_state.session_path = None
    st.session_state.manual_text = ""


def run_coaching(
    *,
    audio_value,
    manual_text: str,
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
        st.warning("Add a little more context so the coaching can produce a specific executive response.")
        return

    with st.spinner("Reframing for executive communication..."):
        coach = build_coach_response(
            client=client,
            transcript=transcript_text,
            audience=audience,
            tone=tone,
            response_length=response_length,
            mode=mode,
        )

    filler_counts = detect_fillers(transcript_text)
    playback_path = OUTPUT_DIR / "executive_playback.mp3"

    with st.spinner("Generating spoken playback..."):
        synthesize_speech(
            client,
            coach.polished_spoken_version,
            playback_path,
            voice=tts_voice,
        )

    session_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "audience": audience,
            "tone": tone,
            "response_length": response_length,
            "mode": mode,
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
    st.session_state.filler_counts = filler_counts
    st.session_state.playback_path = str(playback_path)
    st.session_state.session_path = session_path.name if session_path else None
    st.session_state.manual_text = manual_text


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Executive Voice Copilot</div>
        <div class="hero-subtitle">Turn rough thinking into clear, executive-ready communication</div>
        <div class="hero-note">
            Built for high-stakes leadership conversations, interview prep, and sharper stakeholder communication
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([3, 2])

with top_left:
    st.markdown(
        """
        <div class="section-card">
            <div class="small-label"><strong>How to use it</strong></div>
            Speak or paste a rough thought, then get back a sharper headline, business impact, recommendation,
            challenge question, coaching feedback, and spoken playback
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.info("Demo note: text and audio inputs are processed to generate coaching output.")

with st.sidebar:
    st.header("Session Settings")
    audience = st.selectbox(
        "Audience",
        ["CEO", "CMO", "CFO", "Board", "Peer Leader", "Recruiter"],
    )
    tone = st.selectbox(
        "Tone",
        ["Decisive", "Calm", "Strategic", "Persuasive"],
    )
    response_length = st.selectbox(
        "Response length",
        ["20 seconds", "45 seconds", "90 seconds", "2 minutes"],
    )
    mode = st.selectbox(
        "Mode",
        ["Reframe", "Practice", "Challenge", "Polish"],
    )
    tts_voice = st.selectbox(
        "Voice",
        ["cedar", "marin", "alloy", "ash", "echo", "sage"],
    )
    save_history = st.checkbox("Save session history", value=True)

    st.divider()
    if st.button("Clear session", use_container_width=True):
        clear_session()
        st.rerun()

st.subheader("Try a sample prompt")
sample_cols = st.columns(3)
for idx, (label, prompt) in enumerate(DEFAULT_SAMPLES.items()):
    with sample_cols[idx]:
        st.markdown(
            f'<div class="sample-box"><strong>{label}</strong><br><br>{prompt}</div>',
            unsafe_allow_html=True,
        )
        if st.button(f"Use {label}", key=f"sample_{idx}", use_container_width=True):
            st.session_state.manual_text = prompt

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    audio_value = st.audio_input("Record your answer")
    manual_text = st.text_area(
        "Or paste rough thoughts here",
        height=220,
        value=st.session_state.manual_text,
        placeholder=(
            "Example: I need to explain that our reporting inconsistency is slowing "
            "down decision-making because the team keeps reconciling numbers instead "
            "of acting on the data..."
        ),
    )

    action_cols = st.columns([2, 1])
    with action_cols[0]:
        generate = st.button(
            "Generate executive coaching",
            type="primary",
            use_container_width=True,
        )
    with action_cols[1]:
        if st.button("Clear input", use_container_width=True):
            st.session_state.manual_text = ""
            st.rerun()

if generate:
    run_coaching(
        audio_value=audio_value,
        manual_text=manual_text,
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
    with left:
        st.subheader("Transcript")
        st.write(transcript_text)

        st.subheader("Filler phrases detected")
        if filler_counts:
            for phrase, count in filler_counts.items():
                st.write(f"- {phrase}: {count}")
        else:
            st.write("No common filler phrases detected.")

    with right:
        st.subheader("Audio playback")
        st.audio(playback_path, format="audio/mp3")

        utility_cols = st.columns(2)
        with utility_cols[0]:
            st.download_button(
                "Download polished version",
                data=coach.polished_spoken_version,
                file_name="executive_response.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with utility_cols[1]:
            st.download_button(
                "Download session JSON",
                data=json.dumps(
                    {
                        "transcript": transcript_text,
                        "headline": coach.executive_headline,
                        "business_impact": coach.why_it_matters,
                        "recommendation": coach.recommendation,
                        "spoken_version": coach.polished_spoken_version,
                    },
                    indent=2,
                ),
                file_name="executive_session.json",
                mime="application/json",
                use_container_width=True,
            )

        if session_name:
            st.caption(f"Saved session: {session_name}")

        result_tabs = st.tabs(["Executive Output", "Coaching", "Portfolio View"])

        with result_tabs[0]:
            st.subheader("Executive headline")
            st.write(coach.executive_headline)

            st.subheader("Business impact")
            st.write(coach.why_it_matters)

            st.subheader("Recommendation")
            st.write(coach.recommendation)

            st.subheader("Key supporting points")
            for point in coach.support_points:
                st.write(f"- {point}")

            st.subheader("Polished spoken version")
            st.code(coach.polished_spoken_version, language=None)

            st.subheader("Executive closing line")
            st.write(coach.stronger_closing_line)

        with result_tabs[1]:
            st.subheader("Leadership challenge")
            st.write(coach.tough_question)

            st.subheader("Coaching feedback")
            for item in coach.coaching_feedback:
                st.write(f"- {item}")

            st.subheader("Scores")
            score_cols = st.columns(5)
            score_cols[0].metric("Clarity", coach.scores.clarity)
            score_cols[1].metric("Concision", coach.scores.concision)
            score_cols[2].metric("Presence", coach.scores.executive_presence)
            score_cols[3].metric("Business", coach.scores.business_focus)
            score_cols[4].metric("Action", coach.scores.actionability)

        with result_tabs[2]:
            st.subheader("Portfolio-ready summary")
            st.markdown(
                """
                **Executive Voice Copilot** is an AI-powered communication coach designed to help leaders
                turn rough thinking into concise, executive-ready messaging.

                **Current capabilities**
                - Accepts audio or text input
                - Reframes ideas for executive audiences
                - Produces a sharpened spoken response
                - Generates audio playback
                - Scores clarity, concision, executive presence, business focus, and actionability
                - Saves session history locally

                **Primary use cases**
                - Leadership updates
                - Interview preparation
                - Stakeholder messaging
                - Recommendation framing
                - Executive communication practice
                """
            )
else:
    with right:
        st.subheader("What this app does")
        st.write("- Records voice or accepts rough text")
        st.write("- Transcribes spoken input when audio is used")
        st.write("- Reframes it into an executive answer")
        st.write("- Generates spoken playback")
        st.write("- Scores clarity, concision, executive presence, business focus, and actionability")
        st.write("- Helps users practice sharper leadership communication")

st.divider()
st.caption("Executive Voice Copilot • Demo app for executive communication coaching")