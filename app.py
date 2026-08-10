import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


# -----------------------------
# 1. Define Career Tool
# -----------------------------
@tool
def job_advice(question: str) -> str:
    """
    Provides job-ready preparation advice based on the user's question.
    """

    information = {
        "python": (
            "Learn Python basics, OOP, data structures, algorithms, "
            "and build projects using Flask, Django, Pandas, and APIs."
        ),
        "java": (
            "Learn OOP, collections, exception handling, multithreading, "
            "JDBC, Spring Boot, and practice DSA regularly."
        ),
        "dsa": (
            "Focus on arrays, strings, hashing, stacks, queues, trees, "
            "graphs, and dynamic programming."
        ),
        "interview": (
            "Prepare DSA, DBMS, OS, CN, projects, resume explanation, "
            "and behavioral interview questions."
        ),
        "resume": (
            "Highlight projects, technical skills, internships, certifications, "
            "GitHub links, and measurable achievements."
        )
    }

    question_lower = question.lower()

    for key, value in information.items():
        if key in question_lower:
            return value

    return (
        "Focus on programming fundamentals, DSA, projects, GitHub portfolio, "
        "and communication skills to become job-ready."
    )


tools = [job_advice]


# -----------------------------
# 2. Initialize Model
# -----------------------------
Gemini_API_Key = os.environ.get("Gemini_API_Key")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=Gemini_API_Key,
    temperature=0.3
)


# -----------------------------
# 3. Create Agent
# -----------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a Job-Ready Career Assistant.

Rules:
1. Give concise answers (3-8 lines).
2. When the user asks about job roles after learning a skill, return only:
   - Suitable job roles
   - 1-2 important next skills to learn
3. Do NOT give detailed explanations of each role unless the user explicitly asks.
4. Use bullet points.
5. End with one short suggestion such as "Start with a small project in this area."

Example:
User: What roles can I apply for after learning Python?
Answer:
- Python Developer
- Backend Developer
- Data Analyst
- QA Automation Engineer
- Junior AI/ML Engineer

Next skills: SQL, Git/GitHub, and one framework (FastAPI/Django).
Suggestion: Start with a small Python project and upload it to GitHub.
"""
)


# -----------------------------
# 4. API Input Schema
# -----------------------------
class AgentInput(BaseModel):
    input: str = Field(description="User query")


# -----------------------------
# 5. Helper Functions
# -----------------------------
def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}


def extract_text_response(agent_output: dict) -> str:
    # Find messages
    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if not messages:
        return str(agent_output)

    last = messages[-1]
    content = getattr(last, "content", "")

    # If content is a list, return only the text parts
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)

    return str(content)
# Build runnable chain
formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)


# -----------------------------
# 6. FastAPI App
# -----------------------------
# -----------------------------
# 6. FastAPI App + Simple UI
# -----------------------------
from fastapi.responses import HTMLResponse
from fastapi import Request

app = FastAPI(title="Job-Ready Career Assistant API")


# LangServe API route
add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# Simple Web UI
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Job-Ready Career Assistant</title>
    <style>
        body { font-family: Arial; padding: 40px; background: #f4f6f8; }
        .box { max-width: 700px; margin: auto; background: white; padding: 24px; border-radius: 12px; }
        textarea { width: 100%; height: 120px; padding: 10px; }
        button { margin-top: 10px; padding: 10px 16px; cursor: pointer; }
        #result { margin-top: 20px; padding: 12px; background: #eef2ff; white-space: pre-wrap; }
    </style>
</head>
<body>
<div class="box">
    <h2>Job-Ready Career Assistant</h2>

    <textarea id="question" placeholder="Ask a career question"></textarea><br>

    <button id="askBtn">Ask Assistant</button>

    <div id="result">Your answer will appear here.</div>
</div>

<script>
document.getElementById("askBtn").addEventListener("click", async function () {
    const question = document.getElementById("question").value;

    if (!question.trim()) {
        alert("Please enter a question");
        return;
    }

    document.getElementById("result").innerText = "Thinking...";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ input: question })
        });

        const data = await response.json();
        document.getElementById("result").innerText = data.response || "No response received.";
    } catch (err) {
        document.getElementById("result").innerText = "Error: " + err;
    }
});
</script>
</body>
</html>
"""
   

  


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


@app.post("/ask")
async def ask(request: Request):
    body = await request.json()

    result = formatted_agent_chain.invoke(
        {"input": body.get("input", "")}
    )

    return {"response": result}


@app.get("/health")
def health():
    return {"status": "healthy"}




# -----------------------------
# 7. Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
