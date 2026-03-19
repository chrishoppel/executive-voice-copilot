# Executive Voice Copilot

A Streamlit app that helps you turn rough spoken thoughts into concise, leadership-ready communication.

## What it does

- Accepts microphone input with Streamlit audio recording
- Transcribes audio with OpenAI speech-to-text
- Reframes your thinking into a sharper executive response
- Generates AI voice playback for the polished answer
- Scores clarity, concision, executive presence, business focus, and actionability
- Saves session history locally as JSON

## 1) Open the project in VS Code

- Download or copy this project folder to your computer
- Open VS Code
- Click **File > Open Folder**
- Select `executive_voice_copilot`

## 2) Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

## 4) Add your API key

- Copy `.env.example` to `.env`
- Add your OpenAI API key to `.env`

Example:

```env
OPENAI_API_KEY=your_real_key_here
```

## 5) Run the app

```bash
streamlit run app.py
```

## 6) Use the app

- Choose your audience, tone, response length, mode, and voice in the sidebar
- Record your answer or paste rough text
- Click **Generate executive coaching**
- Review the transcript, executive rewrite, coaching feedback, scores, and audio playback

## Project structure

```text
executive_voice_copilot/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
├── outputs/
└── src/
    ├── audio.py
    ├── coaching.py
    ├── models.py
    └── prompts.py
```

## Next upgrades

- Add challenge mode with a second round follow-up question
- Add progress dashboard and trend charts
- Add PostgreSQL storage instead of local JSON
- Add realtime voice mode later
