# main.py

import json
import requests
import numpy as np
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, UploadFile, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM

# ---------------------------------------------------------------
# FULL CORS + ACCESS CONTROL
# ---------------------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BabySisting API (Groq + ElevenLabs)")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",  # allow everything (optional)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add access-control headers to every outgoing response
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# ============================================================
# 1. GROQ CONFIG
# ============================================================

GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


class GroqLLM(LLM):
    api_key: str = GROQ_API_KEY
    model: str = GROQ_MODEL

    @property
    def _llm_type(self) -> str:
        return "groq_custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        res = requests.post(GROQ_URL, json=payload, headers=headers)

        if res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Groq API Error: {res.text}")

        return res.json()["choices"][0]["message"]["content"]


llm = GroqLLM()


# ============================================================
# 2. FAST TF-IDF RETRIEVER
# ============================================================

class FastRetriever:
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = 3):
        q_vec = self.vectorizer.transform([query])
        scores = (q_vec @ self.matrix.T).toarray()[0]
        idx = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in idx if scores[i] > 0.03]


# ============================================================
# 3. LOAD FAQ
# ============================================================

with open("faq.json", "r") as f:
    FAQ: Dict[str, str] = json.load(f)

documents = list(FAQ.values())
retriever = FastRetriever(documents)


# ============================================================
# 4. MEMORY
# ============================================================

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question"
)


# ============================================================
# 5. PROMPT TEMPLATE
# ============================================================

template = """
You are a helpful customer service assistant.

Use:
- Retrieved context
- Conversation history

If unsure, respond EXACTLY:
"I'm not fully sure about this. Please contact the service provider for more assistance."

Context:
{context}

Chat History:
{chat_history}

User Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)


# ============================================================
# 6. MODELS
# ============================================================

class ChatRequest(BaseModel):
    user_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    answer: str


# ============================================================
# 7. /chat — TEXT CHAT ENDPOINT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.message

    ctx_docs = retriever.search(question)
    context = "\n".join(ctx_docs) if ctx_docs else "No relevant FAQ found."

    answer = chain.run(context=context, question=question)

    if any(x in answer.lower() for x in ["not sure", "don't know", "contact"]):
        answer = "I'm not fully sure about this. Please contact the service provider for more assistance."

    return ChatResponse(answer=answer.strip())


# ============================================================
# 8. TTS (TEXT → AUDIO)
# ============================================================

ELEVEN_API_KEY = "sk_aa63d4c3c133bd1de19c4f6c01406cc856867bf3131595f6"
ELEVEN_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
def text_to_speech(req: TTSRequest):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"

    payload = {
        "text": req.text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
    }

    res = requests.post(
        url,
        headers={
            "Accept": "audio/mpeg",
            "xi-api-key": ELEVEN_API_KEY,
            "Content-Type": "application/json"
        },
        json=payload,
        stream=True
    )

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=res.text)

    return StreamingResponse(res.iter_content(chunk_size=1024), media_type="audio/mpeg")


# ============================================================
# 9. STT (AUDIO → TEXT)
# ============================================================

@app.post("/stt")
def speech_to_text(file: UploadFile):
    res = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": ELEVEN_API_KEY},
        files={"file": (file.filename, file.file, file.content_type)},
        data={"model_id": "scribe_v2"}
    )

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=res.text)

    return {"text": res.json().get("text", "")}


# ============================================================
# 10. FULL VOICE CHAT PIPELINE (STT → LLM → TTS)
# ============================================================

@app.post("/voice-chat")
def voice_chat(file: UploadFile):
    # ========== 1. STT ==========
    stt_res = requests.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": ELEVEN_API_KEY},
        files={"file": (file.filename, file.file, file.content_type)},
        data={"model_id": "scribe_v2"}
    )

    if stt_res.status_code != 200:
        raise HTTPException(status_code=500, detail=stt_res.text)

    user_text = stt_res.json().get("text", "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Speech transcription failed.")

    # ========== 2. LLM ==========
    ctx_docs = retriever.search(user_text)
    context = "\n".join(ctx_docs) if ctx_docs else "No relevant FAQ found."
    answer_text = chain.run(context=context, question=user_text)

    # ========== 3. TTS ==========
    tts_res = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}",
        headers={
            "Accept": "audio/mpeg",
            "xi-api-key": ELEVEN_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "text": answer_text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}
        },
        stream=True
    )

    if tts_res.status_code != 200:
        raise HTTPException(status_code=500, detail=tts_res.text)

    return StreamingResponse(tts_res.iter_content(chunk_size=1024), media_type="audio/mpeg")
