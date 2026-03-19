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

st.set_page_config(page_title="Executive Voice Copilot", layout="wide")

st.title("Executive Voice Copilot")
st.caption("Speak your rough thought. Get back a sharper, more executive answer.")

if not api_key:
    st.error("OPENAI_API_KEY is missing. Add your real API key to the .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

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

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    audio_value = st.audio_input("Record your answer")
    manual_text = st.text_area(
        "Or paste rough thoughts here",
        height=220,
        placeholder=(
            "Example: I need to explain that our reporting inconsistency is slowing "
            "down decision-making because the team keeps reconciling numbers instead "
            "of acting on the data..."
        ),
    )
    generate = st.button(
        "Generate executive coaching",
        type="primary",
        use_container_width=True,
    )

if generate:
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
        st.stop()

    if len(transcript_text.split()) < 6:
        st.warning("Add a little more context so the coaching can produce a specific executive response.")
        st.stop()

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
        st.audio(str(playback_path), format="audio/mp3")

        if session_path:
            st.caption(f"Saved session: {session_path.name}")

        st.subheader("Executive headline")
        st.write(coach.executive_headline)

        st.subheader("Business impact")
        st.write(coach.why_it_matters)

        st.subheader("Recommendation")
        st.write(coach.recommendation)

        st.subheader("Supporting points")
        for point in coach.support_points:
            st.write(f"- {point}")

        st.subheader("Polished spoken version")
        st.write(coach.polished_spoken_version)

        st.subheader("Stronger closing line")
        st.write(coach.stronger_closing_line)

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

else:
    with right:
        st.subheader("What this app does")
        st.write("- Records voice or accepts rough text")
        st.write("- Transcribes spoken input when audio is used")
        st.write("- Reframes it into an executive answer")
        st.write("- Generates spoken playback")
        st.write("- Scores clarity, concision, executive presence, business focus, and actionability")