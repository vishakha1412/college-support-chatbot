from fastapi import FastAPI,UploadFile, Form
from pydantic import BaseModel
from college_assistant import graph,build_retriever
from fastapi.middleware.cors import CORSMiddleware
 
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    programme: str = "B.Tech"
    friend_mode: bool = True


conversation_history = []

@app.post("/chat")
async def chat_endpoint(
    query: str = Form(...),
    programme: str = Form("B.Tech"),
    friend_mode: bool = Form(True),
    academic_pdf: UploadFile | None = None,
    fee_pdf: UploadFile | None = None
):
    
    global academic_retriever, fees_retriever,conversation_history
    if academic_pdf:
        with open("academic.pdf", "wb") as f:
            f.write(await academic_pdf.read())
        academic_retriever = build_retriever("academic.pdf")

    if fee_pdf:
        with open("fee.pdf", "wb") as f:
            f.write(await fee_pdf.read())
        fees_retriever = build_retriever("fee.pdf")

    conversation_history.append(("human", query))
    config = {"configurable": {"thread_id": "student-session-1"}}
    result = graph.invoke(
        {
            "programme": programme,
            "friend_mode": friend_mode,
            "messages":conversation_history
        },
        config=config
    )
    ai_msg = result["messages"][-1].content
    conversation_history.append(("ai", ai_msg))

    return {"answer": ai_msg}