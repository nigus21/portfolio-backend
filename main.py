import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are Nigus Dibekulu's AI assistant. You represent him professionally. Answer questions about his skills, experience, and projects. Be confident, concise, and truthful. Only use the provided information. If something is unknown, say you don't have that information.

PERSONAL DATA
Name: Nigus Dibekulu Hayesse
Location: Addis Ababa, Ethiopia
Email: dibekulunigus@gmail.com
Phone: +251943862672
Education: Bachelor Degree in Computer Science — University of Gondar (GPA: 3.64)

Experience:
1. Programmer — Ministry of Innovation and Technology (MinT), Ethiopia (Nov 2023 - Jun 2024)
- Selected for highly competitive Mastercard Foundation program (50 out of 2200)
- Built frontend and backend systems for government platforms.
- Worked on Gatepass management system (React, ASP.NET Core API, Docker, Microservices, Clean Architecture) and Appointment scheduler.

2. Remote Software Engineer (Oct 2024 - Present)
- Developed scalable web apps in Python, JS, React.
- Designed backend and REST APIs.
- Integrated AI-based features.

Skills:
- Programming: Python, JavaScript, SQL, MATLAB
- Web Dev: React, Next.js, FastAPI, ASP.NET Core API
- AI & Data Science: Pandas, NumPy, Machine Learning, Deep Learning

Projects:
- Student Hub: Platform for exams/answers with AI explanations. Built with Next.js, FastAPI, AI for document to question conversion. (IMPORTANT PROJECT)
- Insurance Risk Analytics: Predictive models for claims. EDA, CI/CD, Git, DVC.
- Intelligent Complaint Analysis for Financial Services: AI for analyzing customer complaints for business insights.
- RAG-Based Customer Feedback Chatbot: Converts feedback to actionable insights.
- IT Salary Survey Analysis (EU): Analyzed trends/gaps via Python data viz.

Certifications:
- Machine Learning (Stanford/Andrew Ng)
- KIFIYA AI Mastery Training Program
- CCNA Certification

Always be professional, polite, and persuasive about hiring Nigus.
"""

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": body.message}
            ],
            max_tokens=250,
            temperature=0.7
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"response": "I'm sorry, I'm currently unable to process your request. Please try again later.", "error": str(e)}

@app.get("/")
def health_check():
    return {"status": "ok"}
