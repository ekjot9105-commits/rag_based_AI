import json
import os
import sys
import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
JOBLIB_FILE = os.path.join(BASE_DIR, "embeddings.joblib")


def check_ollama_server():
    """Verify that Ollama server is running and accessible."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def create_embeddings(text_list, model=EMBED_MODEL):
    """Generate embeddings for text chunks using Ollama."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": model,
                "input": text_list
            },
            timeout=60
        )
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Error: Cannot connect to Ollama at {OLLAMA_URL}.")
        print("   Please start Ollama with: 'ollama serve'")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request Error while generating embeddings: {e}")
        return None

    data = r.json()
    if "error" in data:
        print(f"\n❌ Ollama API Error: {data['error']}")
        return None

    if "embeddings" in data:
        return data["embeddings"]
    if "embedding" in data:
        return [data["embedding"]]

    print("⚠️ Unexpected response format from Ollama:", data)
    return None


def inference(prompt, model=LLM_MODEL):
    """Run generation inference with Ollama."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Error: Cannot connect to Ollama at {OLLAMA_URL}.")
        print("   Please start Ollama with: 'ollama serve'")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request Error during inference: {e}")
        return None

    response = r.json()
    if "error" in response:
        print(f"\n❌ Ollama Inference Error: {response['error']}")
        return None

    return response.get("response", "")


def format_seconds(seconds):
    """Format seconds into MM:SS string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def answer_query(query, df, top_k=5):
    """Find top similar chunks and ask the LLM."""
    print(f"\n🔍 Searching for: \"{query}\"")
    embeddings = create_embeddings([query])
    if embeddings is None or len(embeddings) == 0:
        print("❌ Failed to generate embeddings for query.")
        return None

    query_embedding = embeddings[0]

    # Compute cosine similarity across all chunks
    chunk_embeddings = np.vstack(df["embedding"].values)
    similarities = cosine_similarity(chunk_embeddings, [query_embedding]).flatten()

    # Get top_k most similar chunks (using positional iloc)
    top_indices = similarities.argsort()[::-1][:top_k]
    matched_df = df.iloc[top_indices]

    print("\n📌 Top matching video segments:")
    for idx, (_, row) in enumerate(matched_df.iterrows(), start=1):
        sim = similarities[df.index.get_loc(row.name)]
        start_fmt = format_seconds(row['start'])
        end_fmt = format_seconds(row['end'])
        print(f"  {idx}. [Tutorial #{row['number']}] [{start_fmt} - {end_fmt}] (score: {sim:.3f}): {row['text'].strip()}")

    # Prepare prompt for LLM
    context_json = matched_df[["number", "start", "end", "text"]].to_json(orient="records")
    prompt = f"""I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{context_json}
---------------------------------
"{query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
"""

    prompt_file = os.path.join(BASE_DIR, "prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    print("\n🤖 Generating answer with LLM...")
    answer = inference(prompt)
    if answer is None:
        return None

    response_file = os.path.join(BASE_DIR, "response.txt")
    with open(response_file, "w", encoding="utf-8") as f:
        f.write(answer)

    print("\n========================= ANSWER =========================")
    print(answer)
    print("==========================================================")
    print(f"📝 Prompt saved to: {prompt_file}")
    print(f"📝 Response saved to: {response_file}\n")
    return answer


def main():
    if not check_ollama_server():
        print(f"❌ Error: Ollama server is not running at {OLLAMA_URL}.")
        print("   Please start Ollama server (e.g. 'ollama serve') before running this script.")
        sys.exit(1)

    if not os.path.exists(JOBLIB_FILE):
        print(f"❌ Error: '{JOBLIB_FILE}' not found.")
        print("   Please run 'python data_frame.py' to generate embeddings.joblib first.")
        sys.exit(1)

    print(f"📂 Loading database from {JOBLIB_FILE}...")
    df = joblib.load(JOBLIB_FILE)
    print(f"✅ Loaded {len(df)} chunks into memory.")

    # Check if query passed via command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        answer_query(query, df)
        return

    # Interactive mode
    print("\n💡 Interactive mode (type 'exit' or 'quit' to stop):")
    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break
            answer_query(query, df)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
