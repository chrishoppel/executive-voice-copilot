from pathlib import Path
from openai import OpenAI


def transcribe_audio(client: OpenAI, audio_path: Path) -> str:
    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
        )
    return transcript.text.strip()


def synthesize_speech(client: OpenAI, text: str, output_path: Path, voice: str = "cedar") -> Path:
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions="Speak in a calm, direct, confident executive coaching style.",
    ) as response:
        response.stream_to_file(output_path)
    return output_path
