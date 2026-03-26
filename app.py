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
        padding-top: 0.9rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 1.2rem 1.3rem;
        border-radius: 20px;
        margin-bottom: 0.9rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        opacity: 0.94;
        margin-bottom: 0.35rem;
    }
    .hero-note {
        font-size: 0.9rem;
        opacity: 0.82;
    }
    .panel {
        background: #f8fafc;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }
    .panel-title {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.45rem;
    }
    .sample-chip {
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #dbe3ee;
        border-radius: 14px;
        padding: 0.7rem 0.8rem;
        margin-bottom: 0.55rem;
        min-height: 110px;
        font-size: 0.92rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.6rem;
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

SITUATION_OPTIONS = [
    "Executive update",
    "Interview answer",
    "Recommendation to leadership",
    "Difficult stakeholder conversation",
    "Board / CFO challenge",
]

SAMPLE_PROMPTS = {
    "Executive update": (
        "I need to explain to leadership that reporting inconsistency is slowing decisions "
        "because the team keeps reconciling numbers instead of acting on the data"
    ),
    "Interview answer": (
        "I need to explain how I lead analytics teams in a way that improves decision quality, "
        "not just reporting output"
    ),
    "Recommendation to leadership": (
        "I need to explain to the CFO that we should not cut analytics resources right now "
        "because reducing capacity will slow reporting, weaken decision support, and hurt spend optimization"
    ),
    "Difficult stakeholder conversation": (
        "I need to tell a cross-functional leader that the current request is not scoped well enough "
        "to deliver a reliable answer quickly"
    ),
    "Board / CFO challenge": (
        "I need to explain why fragmented customer data is limiting personalization and reducing marketing efficiency"
    ),
}

STATE_DEFAULTS = {
    "transcript_text": "",
    "coach": None,
    "filler_counts": {},
    "playback_path": None,
    "session_path": None,
    "manual_text": "",
    "challenge_input": "",
    "challenge_result": None,
    "selected_situation": SITUATION_OPTIONS[0],
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_session() -> None:
    st.session_state.transcript_text = ""
    st.session_state.coach = None
    st.session_state.filler_counts = {}
    st.session_state.playback_path = None
    st.session_state.session_path = None
    st.session_state.manual_text = ""
    st.session_state.challenge_input = ""
    st.session_state.challenge_result = None


def build_why_stronger_points(coach, transcript_text: str) -> list[str]:
    points = []

    if transcript_text:
        points.append("It leads with the main point instead of building up to it")

    if coach.why_it_matters:
        points.append("It makes the business implication clearer and easier to act on")

    if coach.recommendation:
        points.append("It gives a more direct point of view instead of staying descriptive")

    if coach.stronger_closing_line:
        points.append("It ends with a cleaner, more confident takeaway")

    return points[:3]


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
    situation: str,
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

    coaching_mode = mode
    if situation == "Board / CFO challenge" and mode == "Practice":
        coaching_mode = "Challenge"

    with st.spinner("Sharpening your answer..."):
        coach = build_coach_response(
            client=client,
            transcript=transcript_text,
            audience=audience,
            tone=tone,
            response_length=response_length,
            mode=coaching_mode,
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
            "mode": coaching_mode,
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
    st.session_state.challenge_input = ""
    st.session_state.challenge_result = None


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Executive Communication Coach</div>
        <div class="hero-subtitle">Practice turning rough thoughts into clear, confident, leadership-ready communication</div>
        <div class="hero-note">Built for executive updates, interviews, stakeholder conversations, and recommendation framing</div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([3, 2])

with top_left:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">How it works</div>
            <div>1. Choose the situation you are preparing for</div>
            <div>2. Record or paste your rough answer</div>
            <div>3. Get a sharper version and clear coaching</div>
            <div>4. Practice again using the challenge question</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.info("This is a coaching product. Use it to practice, compare, and improve how you communicate over time.")

with st.sidebar:
    st.header("Practice Setup")

    selected_situation = st.selectbox(
        "Situation",
        SITUATION_OPTIONS,
        index=SITUATION_OPTIONS.index(st.session_state.selected_situation),
    )
    st.session_state.selected_situation = selected_situation

    audience = st.selectbox(
        "Audience",
        ["CEO", "CMO", "CFO", "Board", "Peer Leader", "Recruiter"],
    )
    tone = st.selectbox(
        "Tone",
        ["Decisive", "Calm", "Strategic", "Persuasive"],
    )
    response_length = st.selectbox(
        "Answer length",
        ["20 seconds", "45 seconds", "90 seconds", "2 minutes"],
    )
    mode = st.selectbox(
        "Practice mode",
        ["Practice", "Reframe", "Challenge", "Polish"],
    )
    tts_voice = st.selectbox(
        "Playback voice",
        ["cedar", "marin", "alloy", "ash", "echo", "sage"],
    )
    save_history = st.checkbox("Save session history", value=True)

    st.divider()
    if st.button("Clear session", use_container_width=True):
        clear_session()
        st.rerun()

st.subheader("Start with a situation")
situation_cols = st.columns(3)

for idx, situation in enumerate(SITUATION_OPTIONS[:3]):
    with situation_cols[idx]:
        st.markdown(
            f"""
            <div class="sample-chip">
                <strong>{situation}</strong><br><br>
                {SAMPLE_PROMPTS[situation]}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Use {situation}", key=f"situation_{idx}", use_container_width=True):
            st.session_state.selected_situation = situation
            st.session_state.manual_text = SAMPLE_PROMPTS[situation]
            st.rerun()

more_cols = st.columns(2)
for idx, situation in enumerate(SITUATION_OPTIONS[3:], start=3):
    with more_cols[idx - 3]:
        st.markdown(
            f"""
            <div class="sample-chip">
                <strong>{situation}</strong><br><br>
                {SAMPLE_PROMPTS[situation]}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Use {situation}", key=f"situation_{idx}", use_container_width=True):
            st.session_state.selected_situation = situation
            st.session_state.manual_text = SAMPLE_PROMPTS[situation]
            st.rerun()

left, right = st.columns([1, 1])

with left:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Your attempt</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    audio_value = st.audio_input("Record your answer")

    manual_text = st.text_area(
        "Or paste your rough answer",
        height=180,
        value=st.session_state.manual_text,
        placeholder="Type the way you would actually say it, even if it feels rough or incomplete...",
    )

    action_cols = st.columns([2, 1])
    with action_cols[0]:
        generate = st.button(
            "Sharpen my answer",
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
        situation=selected_situation,
    )

coach = st.session_state.coach
transcript_text = st.session_state.transcript_text
filler_counts = st.session_state.filler_counts
playback_path = st.session_state.playback_path
session_name = st.session_state.session_path

if coach:
    why_stronger = build_why_stronger_points(coach, transcript_text)

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Sharper answer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Audio playback")
        st.audio(playback_path, format="audio/mp3")

        st.subheader("Sharpened version")
        st.code(coach.polished_spoken_version, language=None)

        utility_cols = st.columns(2)
        with utility_cols[0]:
            st.download_button(
                "Download sharpened answer",
                data=coach.polished_spoken_version,
                file_name="sharpened_answer.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with utility_cols[1]:
            st.download_button(
                "Download session JSON",
                data=json.dumps(
                    {
                        "situation": selected_situation,
                        "transcript": transcript_text,
                        "headline": coach.executive_headline,
                        "business_impact": coach.why_it_matters,
                        "recommendation": coach.recommendation,
                        "spoken_version": coach.polished_spoken_version,
                        "challenge": coach.tough_question,
                    },
                    indent=2,
                ),
                file_name="communication_coaching_session.json",
                mime="application/json",
                use_container_width=True,
            )

        if session_name:
            st.caption(f"Saved session: {session_name}")

        st.subheader("Why this is stronger")
        for point in why_stronger:
            st.write(f"- {point}")

        st.subheader("Coach's challenge")
        st.write(coach.tough_question)

        st.subheader("One thing to improve next")
        if coach.coaching_feedback:
            st.write(coach.coaching_feedback[0])
        else:
            st.write("Make the main point earlier and keep the recommendation direct.")

    below_left, below_right = st.columns([1, 1])

    with below_left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Compare</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Your original answer")
        st.write(transcript_text)

        st.subheader("Headline")
        st.write(coach.executive_headline)

        st.subheader("Business impact")
        st.write(coach.why_it_matters)

        st.subheader("Recommendation")
        st.write(coach.recommendation)

        st.subheader("Key supporting points")
        for point in coach.support_points:
            st.write(f"- {point}")

        st.subheader("Filler phrases detected")
        if filler_counts:
            for phrase, count in filler_counts.items():
                st.write(f"- {phrase}: {count}")
        else:
            st.write("No common filler phrases detected.")

    with below_right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Practice again</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("Answer the challenge question in your own words, then compare how you improved.")

        challenge_input = st.text_area(
            "Your response to the challenge",
            height=150,
            value=st.session_state.challenge_input,
            placeholder="Type how you would answer the challenge question...",
        )

        retry_cols = st.columns([2, 1])
        with retry_cols[0]:
            retry = st.button("Sharpen my challenge response", use_container_width=True)
        with retry_cols[1]:
            if st.button("Reset practice", use_container_width=True):
                st.session_state.challenge_input = ""
                st.session_state.challenge_result = None
                st.rerun()

        if retry:
            st.session_state.challenge_input = challenge_input
            if len(challenge_input.split()) < 6:
                st.warning("Add a little more detail so the challenge response can be sharpened.")
            else:
                with st.spinner("Sharpening challenge response..."):
                    challenge_result = build_coach_response(
                        client=client,
                        transcript=challenge_input,
                        audience=audience,
                        tone=tone,
                        response_length=response_length,
                        mode="Polish",
                    )
                st.session_state.challenge_result = challenge_result

        if st.session_state.challenge_result:
            retry_result = st.session_state.challenge_result

            st.subheader("Sharpened challenge response")
            st.code(retry_result.polished_spoken_version, language=None)

            st.subheader("What improved")
            if retry_result.coaching_feedback:
                for item in retry_result.coaching_feedback[:2]:
                    st.write(f"- {item}")

    st.divider()
    score_cols = st.columns(5)
    score_cols[0].metric("Clarity", coach.scores.clarity)
    score_cols[1].metric("Concision", coach.scores.concision)
    score_cols[2].metric("Presence", coach.scores.executive_presence)
    score_cols[3].metric("Business", coach.scores.business_focus)
    score_cols[4].metric("Action", coach.scores.actionability)

else:
    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">What you will get</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("- A sharper version of your answer")
        st.write("- Clear feedback on why it improved")
        st.write("- One realistic challenge question")
        st.write("- A chance to practice again immediately")
        st.write("- Audio playback for delivery practice")

st.divider()
st.markdown(
    '<div class="footer-note">Executive Communication Coach • Practice clear, confident communication over time</div>',
    unsafe_allow_html=True,
)