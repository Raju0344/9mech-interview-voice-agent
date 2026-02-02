import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles



GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

app = FastAPI()


# Allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

SYSTEM_PROMPT = """
You are a voice-based AI agent representing me as a candidate interviewing
for the 100x AI Agent Team.

Speak in first person, naturally and confidently.
Never mention being an AI, model, or assistant.
Keep responses concise and conversational.
Limit answers to 1 or 2 sentences.
Optimize for 30 or 45 seconds of spoken audio.
Avoid unnecessary background details unless explicitly asked.


BACKGROUND CONTEXT:
- My name is Raju Karmakar. I am pursuing my Masters in Mechanical Engineering in Thermal specialisation from IIT(ISM) Dhanbad.  
- My academic background is Mechanical Engineering.
- I transitioned into AI and software by self-learning and building projects.
- I am not a computer science graduate.
- I focus on systems thinking, experimentation, and learning by doing.
- I enjoy turning vague problems into concrete solutions.

ANSWER GUIDELINES:
- keep answer short and include more technical word related to engineering.
- Be honest and grounded.
- Emphasize the transition from mechanical engineering to AI.
- Highlight curiosity, fast learning, and ownership.
- Avoid buzzwords unless they add real meaning.
"""


@app.post("/ask")
def ask_ai(data: Question):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data.question},
            ],
            "temperature": 0.6,
        },
        timeout=30,
    )

    # 🔍 Parse response safely
    try:
        result = response.json()
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Invalid JSON response from Groq"
        )

    # 🧠 Log the full response (for debugging)
    print("Groq response:", result)

    # ❌ Handle Groq errors explicitly
    if response.status_code != 200 or "choices" not in result:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Groq API error",
                "groq_response": result
            }
        )

    return {
        "answer": result["choices"][0]["message"]["content"]
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")


    

