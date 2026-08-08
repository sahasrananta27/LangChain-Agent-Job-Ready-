from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import os

# ---------------------------------------------------
# Configure Gemini API Key
# Set this in Render -> Environment Variables
# GOOGLE_API_KEY = your_key
# ---------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

app = FastAPI(title="Career Placement Agent")

# ---------------------------------------------------
# Tool 1: Resume Skill Analyzer
# ---------------------------------------------------

@tool
def analyze_resume_skills(resume_text: str) -> str:
    """
    Analyze resume text and identify technical skills.
    """
    skills = []

    keywords = [
        "python", "java", "sql", "mysql", "pandas",
        "numpy", "machine learning", "langchain",
        "langgraph", "fastapi", "flask", "git",
        "github", "docker", "api", "javascript"
    ]

    text = resume_text.lower()

    for k in keywords:
        if k in text:
            skills.append(k)

    if not skills:
        return "No technical skills detected."

    return "Detected skills: " + ", ".join(skills)

# ---------------------------------------------------
# Tool 2: Job Roadmap Generator
# ---------------------------------------------------

@tool
def job_roadmap(role: str) -> str:
    """
    Generate a learning roadmap for a target job role.
    """
    role = role.lower()

    roadmaps = {
        "ai engineer": "Python -> Machine Learning -> Deep Learning -> LangChain -> LangGraph -> RAG -> MLOps -> Deployment",
        "data scientist": "Python -> Statistics -> Pandas -> SQL -> Machine Learning -> Visualization -> Projects",
        "software engineer": "DSA -> Java/Python -> OOP -> DBMS -> APIs -> GitHub -> System Design",
        "backend developer": "Python/Java -> FastAPI/Spring -> SQL -> REST APIs -> Authentication -> Deployment"
    }

    return roadmaps.get(role, "Build strong programming, DBMS, DSA, projects, and deployment skills.")

# ---------------------------------------------------
# Tool 3: Interview Question Generator
# ---------------------------------------------------

@tool
def interview_questions(role: str) -> str:
    """
    Generate sample interview questions.
    """
    prompt = f"""
    Generate 5 technical interview questions for a {role}.
    Keep them concise and suitable for campus placements.
    """

    response = llm.invoke(prompt)
    return response.content

# ---------------------------------------------------
# Create Agent
# ---------------------------------------------------

agent = create_agent(
    model=llm,
    tools=[analyze_resume_skills, job_roadmap, interview_questions],
    system_prompt="""
You are a professional career placement assistant.
Help students prepare for internships and placements.
Use tools whenever appropriate.
Give concise, actionable answers.
"""
)

# ---------------------------------------------------
# Request Model
# ---------------------------------------------------

class Query(BaseModel):
    message: str

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Career Placement Agent is running",
        "docs": "/docs"
    }

@app.post("/chat")
def chat(query: Query):
    result = agent.invoke({
        "messages": [HumanMessage(content=query.message)]
    })

    final_message = result["messages"][-1].content

    return {"response": final_message}
