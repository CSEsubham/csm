#%%
import json
import requests
import numpy as np
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM


# =============================
# 1. GROQ CONFIG
# =============================
GROQ_API_KEY = ""
#GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
#GROQ_MODEL = "llama-3.1-8b-instant"


# =============================
# 2. GROQ LLM WRAPPER
# =============================
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


# =============================
# 3. FAST TF-IDF RETRIEVER
# =============================
class FastRetriever:
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = 3):
        q_vec = self.vectorizer.transform([query])
        scores = (q_vec @ self.matrix.T).toarray()[0]

        idx = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in idx if scores[i] > 0.03]   # threshold


# =============================
# 4. LOAD FAQ
# =============================
with open("faq.json", "r") as f:
    FAQ: Dict[str, str] = json.load(f)

documents = list(FAQ.values())
retriever = FastRetriever(documents)


# =============================
# 5. MEMORY
# =============================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question"
)


# =============================
# 6. PROMPT TEMPLATE
# =============================
template = """
You are a polite customer service assistant.

Use:
- Retrieved context
- Full conversation history

If unsure or missing context, respond exactly:
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


# =============================
# 7. FASTAPI SETUP
# =============================
app = FastAPI(title="Groq Customer Service Bot + TTS")


class ChatRequest(BaseModel):
    user_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    answer: str


# =============================
# 8. /chat ENDPOINT
# =============================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.message

    # Retrieve context
    ctx_docs = retriever.search(question)
    context = "\n".join(ctx_docs) if ctx_docs else "No relevant FAQ found."

    # LangChain pipeline
    answer = chain.run(context=context, question=question)

    # fallback
    if any(x in answer.lower() for x in ["not sure", "don't know", "contact"]):
        answer = "I'm not fully sure about this. Please contact the service provider for more assistance."

    return ChatResponse(answer=answer.strip())


# =============================
# 9. TTS (ElevenLabs - Bella voice)
# =============================
ELEVEN_API_KEY = ""
#ELEVEN_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Bella
#ELEVEN_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}/stream"


@app.post("/tts")
def text_to_speech(text: str):
    headers = {
        "Accept": "audio/mpeg",
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.8
        }
    }

    res = requests.post(ELEVEN_URL, json=payload, headers=headers, stream=True)

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=res.text)

    return StreamingResponse(
        res.iter_content(chunk_size=1024),
        media_type="audio/mpeg"
    )


# Run with:
# uvicorn main:app --reload --port 8000
