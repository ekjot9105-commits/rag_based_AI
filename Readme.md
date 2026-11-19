# RAG-Based AI Teaching Assistant for Video Courses

An intelligent Retrieval-Augmented Generation (RAG) assistant designed to index video courses, transcribe lectures, generate semantic vector embeddings, and answer student questions with exact timestamps and video references.

---

## 🌟 Overview & Architecture

```
[ Videos (.mp4) ]
       │
       ▼ (Step 1: process_video.py via ffmpeg)
[ Audio Files (.mp3) ]
       │
       ▼ (Step 2: create_chunks.py via OpenAI Whisper)
[ Timestamped Text Chunks (jsons/*.json) ]
       │
       ▼ (Step 3: read_chunks1.py via Ollama nomic-embed-text)
[ Vector Embeddings (embedded_chunks.json) ]
       │
       ▼ (Step 4: data_frame.py via Pandas & Joblib)
[ Embeddings Database (embeddings.joblib) ]
       │
       ▼ (Step 5: process_incoming.py via Cosine Similarity + Ollama Llama 3.2)
[ Accurate Answers with Video Number & Timestamp References ]
```

---

## 📋 Prerequisites

1. **Python 3.10+** (Tested on Python 3.12, 3.13, 3.14)
2. **FFmpeg**: Required for audio extraction.
   - Verify installation: `ffmpeg -version`
   - Windows: Install via `winget install Gyan.FFmpeg` or download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
3. **Ollama**: Required for local embeddings and LLM inference.
   - Download & install from [ollama.com](https://ollama.com/)

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/ekjot9105-commits/rag_based_AI.git
cd rag_based_AI
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull required models with Ollama
Make sure the Ollama server is running (`ollama serve`), then pull the required models:
```bash
# Embedding model
ollama pull nomic-embed-text

# LLM for generation
ollama pull llama3.2
```

---

## 📖 Step-by-Step Pipeline

### Step 1: Collect & Convert Videos to Audio
Place course videos into the `videos/` (or `vedeos/`) folder. Filenames can include a `#<number>` tag (e.g. `Tutorial #3.mp4` or `Topic_Name#3.mp4`).

Run the converter:
```bash
python process_video.py
```
*Outputs: MP3 audio files in the `audios/` folder.*

---

### Step 2: Transcribe Audio to Timestamped Chunks
Transcribe audio files into subtitle chunks using OpenAI Whisper:
```bash
python create_chunks.py
```
*Outputs: Structured JSON files in the `jsons/` folder containing timestamps (`start`, `end`) and transcribed text.*

> **Tip:** You can choose a smaller or larger Whisper model using an environment variable:
> ```bash
> set WHISPER_MODEL=base
> python create_chunks.py
> ```

---

### Step 3: Generate Vector Embeddings
Batch-embed the subtitle chunks using `nomic-embed-text` through Ollama:
```bash
python read_chunks1.py
```
*Outputs: `embedded_chunks.json` containing the chunk metadata alongside 768-dimensional embedding vectors.*

---

### Step 4: Build Searchable Joblib Database
Convert `embedded_chunks.json` into an optimized Pandas DataFrame stored in a Joblib file:
```bash
python data_frame.py
```
*Outputs: `embeddings.joblib`.*

---

### Step 5: Ask Questions to the AI Assistant
Query the assistant either directly via command line or in interactive mode:

**Command-line query:**
```bash
python process_incoming.py "What is an HTML tag?"
```

**Interactive mode:**
```bash
python process_incoming.py
```
Then enter your questions interactively. Type `exit` or `quit` to exit.

The assistant retrieves the top most relevant video segments, formats the timestamps (`MM:SS`), and generates a human-like response guiding you to the exact tutorial and timestamp.

---

## 📁 Repository Structure

```
rag_based_AI/
├── audios/                 # Extracted MP3 audio files
├── jsons/                  # Whisper-transcribed JSON chunks
├── videos/ (or vedeos/)    # Course video files
├── create_chunks.py        # Step 2: Audio transcription & chunking
├── data_frame.py           # Step 4: Converts embedded JSON to embeddings.joblib
├── embedded_chunks.json    # JSON storage of chunks with embedding vectors
├── embeddings.joblib       # Pre-computed DataFrame with embeddings
├── process_incoming.py     # Step 5: Semantic search & LLM response generation
├── process_video.py        # Step 1: Converts videos to MP3 using ffmpeg
├── processvedio.py         # Backward-compatible wrapper for process_video.py
├── read_chunks1.py         # Step 3: Generates embeddings using Ollama
├── stt.py                  # Single-file Whisper transcription utility
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
└── Readme.md               # Project documentation
```

---

## ⚙️ Configuration & Environment Variables

You can customize the models and endpoints by setting environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.2` | Ollama model for answer generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `WHISPER_MODEL` | `large-v2` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v2`) |
