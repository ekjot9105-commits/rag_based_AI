import requests
import os
import json
import sys

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ✅ Create embeddings (batch)
def create_embeddings(text_list, model=EMBED_MODEL):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": model,
                "input": text_list
            },
            timeout=120
        )
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to Ollama at {OLLAMA_URL}. Is Ollama running?")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        return None

    data = r.json()

    # ❌ Handle error in response body
    if "error" in data:
        print("❌ API Error:", data["error"])
        return None

    # ✅ Batch response
    if "embeddings" in data:
        return data["embeddings"]

    # ✅ Fallback (single)
    if "embedding" in data:
        return [data["embedding"]]

    print("⚠️ Unexpected response:", data)
    return None


# ✅ Clean text
def clean_text(text):
    if not text:
        return ""
    return text.strip()


def process_json_chunks():
    if not os.path.exists("jsons"):
        print("❌ Error: 'jsons' folder does not exist.")
        return

    json_files = [f for f in os.listdir("jsons") if f.lower().endswith(".json")]
    if not json_files:
        print("⚠️ No JSON files found in 'jsons' folder.")
        return

    mydict = []
    chunk_id = 0

    for json_file in json_files:
        print(f"\n📂 Processing file: {json_file}")
        file_path = os.path.join("jsons", json_file)

        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        chunks = content.get("chunks", [])
        valid_chunks = []
        texts = []

        # 🔥 Prepare clean data
        for chunk in chunks:
            text = clean_text(chunk.get("text", ""))
            if not text:
                continue
            texts.append(text)
            valid_chunks.append(chunk)

        print("✅ Valid chunks:", len(texts))
        if len(texts) == 0:
            continue

        # 🔥 Batch processing (avoid too large requests)
        BATCH_SIZE = 100
        all_embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            print(f"⚡ Processing batch {i // BATCH_SIZE + 1} of {(len(texts) - 1) // BATCH_SIZE + 1}...")

            embeddings = create_embeddings(batch)
            if embeddings is None:
                print("❌ Batch failed, skipping...")
                continue

            all_embeddings.extend(embeddings)

        print("✅ Total embeddings received:", len(all_embeddings))

        # 🔥 Map embeddings back
        for i, chunk in enumerate(valid_chunks):
            if i >= len(all_embeddings):
                print(f"⚠️ Missing embedding at index {i}")
                break

            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = all_embeddings[i]
            mydict.append(chunk)
            chunk_id += 1

        print(f"✅ Processed chunks so far: {chunk_id}")

    # ✅ SAVE OUTPUT
    output_path = "embedded_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mydict, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 DONE! Saved {len(mydict)} chunks to {output_path}")

if __name__ == "__main__":
    process_json_chunks()
