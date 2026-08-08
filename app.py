from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
import os

app = FastAPI(title="LangChain Job Ready Agent")

# Set API key in Colab:
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

@tool
def resume_tips(role: str) -> str:
    """Give resume tips for a target job role."""
    return f"Add projects, internships, certifications, and skills relevant to {role}."

agent = initialize_agent(
    tools=[resume_tips],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

@app.get("/")
def home():
    return {"message": "LangChain Job Ready Agent"}

@app.get("/career")
def career(role: str):
    result = agent.run(f"Give job readiness advice for {role}")
    return {"response": result}
