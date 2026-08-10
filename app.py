
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
    model="gemma-4-31b-it",
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

Your responsibilities:
1. Suggest suitable job roles based on the user's skills.
2. Answer programming and interview preparation questions.
3. Use the job_advice tool for career preparation queries.
4. Provide practical and beginner-friendly guidance.
5. Recommend next skills to learn and project ideas when useful.
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
    messages = agent_output.get("messages")

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)


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
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 40px;
        }
        .container {
            max-width: 700px;
            margin: auto;
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            font-size: 16px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        button {
            margin-top: 12px;
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background: #1d4ed8;
        }
        #result {
            margin-top: 20px;
            padding: 14px;
            background: #eef2ff;
            border-radius: 8px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Job-Ready Career Assistant</h1>
        <p>Ask about job roles, interview preparation, Python, Java, DSA, or resume guidance.</p>

        <textarea id="question" placeholder="Example: What should I learn for a Java SDE interview?"></textarea>
        <button onclick="askAgent()">Ask Assistant</button>

        <div id="result">Your answer will appear here.</div>
    </div>

<script>
async function askAgent() {
    const question = document.getElementById("question").value;

    const response = await fetch("/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ input: question })
    });

    const data = await response.json();
    document.getElementById("result").innerText = data.response;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        user_input = body.get("input", "")

        result = formatted_agent_chain.invoke({"input": user_input})

        return {"response": result}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}


@app.get("/health")
def health():
    return {"status": "healthy"}




# -----------------------------
# 7. Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
