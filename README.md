# 📚 Notebook LLM API

Upload a document and get an instant **summary**, **Q&A pairs**, and **YouTube resources** — powered by Groq and Gemini.

---

## ✨ What It Does

| Feature | Description |
|---|---|
| 🔍 **Smart Filter** | Validates file type, size, and checks if content is educational |
| 📝 **Summarizer** | Generates a clean summary + 5 key points |
| ❓ **Q&A Generator** | Creates question & answer pairs to test understanding |
| 🎥 **YouTube Search** | Finds related educational videos on YouTube |

---

## 🔑 API Keys — Where to Get Them

You need **3 free API keys** before running the project.

### 1. Groq API Key
> Used for: Smart Filter (classifying content)

1. Go to **[console.groq.com](https://console.groq.com)**
2. Sign up for a free account (no credit card needed)
3. Click **API Keys** in the left sidebar
4. Click **Create API Key**
5. Copy the key → it starts with `gsk_...`

---

### 2. Gemini API Key
> Used for: Summarizer + Q&A Generator

1. Go to **[aistudio.google.com](https://aistudio.google.com)**
2. Sign in with your Google account
3. Click **Get API Key** at the top
4. Click **Create API key in new project**
5. Copy the key → it starts with `AI...`

> ⚠️ **Free tier limits:** 5 requests/minute and 20 requests/day.
> The app handles this automatically with built-in delays.

---

### 3. YouTube Data API Key
> Used for: Finding related YouTube videos

1. Go to **[console.cloud.google.com](https://console.cloud.google.com)**
2. Sign in with your Google account (same as Gemini is fine)
3. Click **Select a project** at the top → **New Project** → give it any name → **Create**
4. In the search bar, type **YouTube Data API v3** → click on it
5. Click **Enable**
6. In the left sidebar go to **APIs & Services → Credentials**
7. Click **+ Create Credentials → API Key**
8. Copy the key → it starts with `AIza...`

> ✅ Free tier gives you **100 search units/day** — more than enough for development.

---

## ⚙️ Setup

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

On Windows if you get an error with `python-magic`, run:
```bash
pip install python-magic-bin
```

### Step 2 — Add your API keys to `config.py`

Open `app/core/config.py` and replace the empty strings with your API keys:

```python
class Settings(BaseSettings):
    GROQ_API_KEY: str = "gsk_your_key_here"
    GEMINI_API_KEY: str = "AI_your_key_here"
    YOUTUBE_API_KEY: str = "AIza_your_key_here"
```

> ⚠️ **Do not share or upload `config.py` to GitHub after adding your keys.**
> Anyone who sees your keys can use your API quota.

### Step 3 — Run the server

```bash
uvicorn app.main:app --reload
```

Open your browser at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to see the interactive API.

---

## 📁 Project Structure

```
application/
├── .env                   ← your API keys (create from .env.example)
├── .env.example           ← template
├── requirements.txt
└── app/
    ├── main.py            ← FastAPI entry point
    ├── api/
    │   └── routes.py      ← API endpoints
    ├── core/
    │   ├── config.py      ← settings & keys
    │   └── schemas.py     ← request/response models
    └── services/
        ├── filter_service.py      ← Groq-powered filter
        ├── extractor_service.py   ← PDF/DOCX/TXT text extraction
        ├── summarizer_service.py  ← Gemini summarizer
        ├── qa_service.py          ← Gemini Q&A generator
        └── youtube_service.py     ← YouTube search
```

---

## 🚀 API Endpoints

### `POST /api/v1/filter`
Validates and classifies the document. Use this to test the filter alone.

**Input:** Upload a file (PDF, DOCX, or TXT)

**Response:**
```json
{
  "status": "approved",
  "message": "Document passed all filters",
  "word_count": 2520,
  "detected_topic": "Embedded Systems",
  "confidence": 0.9
}
```

---

### `POST /api/v1/process`
Full pipeline: filter → summarize → Q&A → YouTube.

**Input:** Upload a file + optional query parameters:

| Parameter | Default | Description |
|---|---|---|
| `include_summary` | `true` | Include summary in response |
| `include_qa` | `true` | Include Q&A pairs |
| `include_youtube` | `true` | Include YouTube videos |
| `num_questions` | `5` | Number of Q&A pairs (1–10) |
| `num_videos` | `5` | Number of YouTube videos (1–10) |

**Example:**
```
POST /api/v1/process?num_questions=7&include_youtube=false
```

**Response:**
```json
{
  "word_count": 2520,
  "detected_topic": "Embedded Systems",
  "confidence": 0.9,
  "summary": {
    "summary": "This lecture covers C programming for embedded systems...",
    "key_points": [
      "C is preferred over Assembly for portability and maintainability",
      "Memory optimization is critical in embedded systems",
      "Bitwise operators are essential for hardware control"
    ]
  },
  "qa": {
    "total": 5,
    "qa_pairs": [
      {
        "question": "Why is C preferred for embedded systems?",
        "answer": "C offers portability across architectures, ease of maintenance..."
      }
    ]
  },
  "youtube": {
    "total": 5,
    "search_query": "Embedded Systems",
    "videos": [
      {
        "title": "Embedded Systems Full Course",
        "url": "https://www.youtube.com/watch?v=...",
        "channel": "...",
        "thumbnail": "...",
        "description": "..."
      }
    ]
  }
}
```

---

## ⚠️ Supported File Types

| Type | Extension | Max Size |
|---|---|---|
| PDF | `.pdf` | 50 MB |
| Word Document | `.docx` | 50 MB |
| Plain Text | `.txt` | 50 MB |

---

## ❌ Why Documents Get Rejected

| Reason | Meaning |
|---|---|
| `invalid_file_type` | File is not PDF, DOCX, or TXT |
| `file_too_large` | File exceeds 50 MB |
| `content_too_short` | Document has fewer than 50 words |
| `not_educational_content` | Content is not educational (e.g. spam, chat logs) |
| `unsafe_content` | Content contains harmful material |

---

## 🛠️ Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Filter | Groq (Llama 3.3 70B) | Free |
| Summarizer | Gemini 2.5 Flash | Free tier |
| Q&A Generator | Gemini 2.5 Flash | Free tier |
| YouTube Search | YouTube Data API v3 | Free tier |
| API Framework | FastAPI | Free |
