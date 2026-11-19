import json
import os
import sys

def transcribe_audios_to_chunks(model_name=None):
    try:
        import whisper
    except ImportError:
        print("❌ Error: 'whisper' is not installed. Please install it using: pip install openai-whisper")
        return

    if not os.path.exists("audios"):
        print("❌ Error: 'audios' directory does not exist.")
        return

    os.makedirs("jsons", exist_ok=True)
    
    if not model_name:
        model_name = os.getenv("WHISPER_MODEL", "large-v2")

    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)

    audio_extensions = (".mp3", ".wav", ".m4a", ".ogg", ".flac")
    audios = [f for f in os.listdir("audios") if f.lower().endswith(audio_extensions)]

    if not audios:
        print("⚠️ No audio files found in 'audios' folder.")
        return

    print(f"Found {len(audios)} audio file(s) to process.")

    for audio in audios:
        out_json_path = os.path.join("jsons", f"{audio}.json")
        if os.path.exists(out_json_path):
            print(f"⏩ Skipping {audio} (already transcribed at {out_json_path})")
            continue

        number = audio.split(".")[0]
        audio_path = os.path.join("audios", audio)
        print(f"🎙️ Transcribing: {audio_path} (Tutorial #{number})...")

        try:
            result = model.transcribe(
                audio=audio_path,
                language="hi",
                task="translate",
                word_timestamps=False
            )

            chunks = []
            for segment in result.get("segments", []):
                chunks.append({
                    "number": number,
                    "start": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "text": segment["text"]
                })

            chunks_with_metadata = {
                "chunks": chunks,
                "text": result.get("text", "")
            }

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(chunks_with_metadata, f, indent=2, ensure_ascii=False)

            print(f"  ✅ Saved {len(chunks)} chunks to {out_json_path}")
        except Exception as e:
            print(f"  ❌ Error transcribing {audio}: {e}")

    print("🎉 Transcription completed!")

if __name__ == "__main__":
    transcribe_audios_to_chunks()
