import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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
    """Provides job-ready preparation advice based on the user's question."""

    information = {
        "python": "Learn Python, OOP, DSA, and build projects with FastAPI or Django.",
        "java": "Learn OOP, collections, JDBC, Spring Boot, and practice DSA.",
        "dsa": "Focus on arrays, strings, stacks, queues, trees, graphs, and DP.",
        "interview": "Prepare DSA, DBMS, OS, CN, projects, and behavioral questions.",
        "resume": "Highlight projects, skills, internships, certifications, and GitHub."
    }

    question_lower = question.lower()

    for key, value in information.items():
        if key in question_lower:
            return value

    return "Focus on programming fundamentals, projects, GitHub, and communication skills."


tools = [job_advice]


# -----------------------------
# 2. Initialize Model
# -----------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
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
2. Suggest suitable job roles.
3. Mention 1-2 next skills to learn.
4. Do not give long explanations unless asked.
5. End with one short suggestion.
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

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if not messages:
        return str(agent_output)

    last = messages[-1]
    content = getattr(last, "content", "")

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)

    return str(content)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)


# -----------------------------
# 6. FastAPI App
# -----------------------------
app = FastAPI(title="Job-Ready Career Assistant API")

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default",
    enabled_endpoints=["invoke"]
)


# -----------------------------
# 7. Simple Web UI
# -----------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Job-Ready Career Assistant</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            padding: 40px;
        }
        .box {
            max-width: 700px;
            margin: auto;
            background: white;
            padding: 24px;
            border-radius: 12px;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            margin-top: 10px;
        }
        button {
            margin-top: 12px;
            padding: 10px 18px;
            cursor: pointer;
        }
        #result {
            margin-top: 20px;
            padding: 12px;
            background: #eef2ff;
            white-space: pre-wrap;
            border-radius: 8px;
        }
    </style>
</head>
<body>
<div class="box">
    <h2>Job-Ready Career Assistant</h2>

    <textarea id="question"
        placeholder="Example: What roles can I apply for after learning Python?"></textarea><br>

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
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ input: question })
        });

        const data = await response.json();
        document.getElementById("result").innerText =
            data.response || "No response received.";

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
    try:
        body = await request.json()
        user_input = body.get("input", "")

        result = formatted_agent_chain.invoke({"input": user_input})

        return {"response": result}

    except Exception as e:
        print("ERROR:", e)
        return {"response": f"Server error: {str(e)}"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# -----------------------------
# 8. Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
