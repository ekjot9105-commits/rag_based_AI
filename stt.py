import json
import os
import sys

def run_stt():
    try:
        import whisper
    except ImportError:
        print("❌ Error: 'whisper' is not installed. Please install it using: pip install openai-whisper")
        return

    audio_path = os.path.join("audios", "output.mp3")
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file '{audio_path}' not found.")
        return

    model_name = os.getenv("WHISPER_MODEL", "large-v2")
    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)

    print(f"🎙️ Transcribing {audio_path}...")
    result = model.transcribe(
        audio=audio_path,
        language="hi",
        task="translate",
        word_timestamps=False
    )

    chunks = []
    for segment in result.get("segments", []):
        chunks.append({
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "text": segment["text"]
        })

    print(f"✅ Transcribed {len(chunks)} chunks.")
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print("📁 Saved result to output.json")

if __name__ == "__main__":
    run_stt()
