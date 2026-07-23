import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from college_assistant import State
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")       
tavily_api_key = os.getenv("TAVILY_API_KEY")  
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

 
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
 
search_tool = TavilySearch(max_results=3)

academic_retriever = None
fees_retriever = None


 
def build_retriever(path: str):
    """
    Loads a PDF, splits it into chunks, and builds a retriever
    out of it using FAISS.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Couldn't find '{path}'. Please put the PDF in the project folder."
        )

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embedding=embeddings)

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 50}
    )

def classification_node(state: State) -> dict:
    """
    Figures out whether the question is about academics, fees, or
    something general. We keep the LLM call isolated here instead
    of putting it inside the router, since LangGraph routers should
    stay lightweight and just read a value, not call an LLM.
    """
    user_query = state["messages"][-1].content

    prompt = f"""
You are a routing and intent-classification agent for a college assistant.

Classify the following user query into exactly ONE of these categories:

1. academic
Use for:
- Courses, subjects, syllabus, faculty, timetable, attendance
- Examinations, results, assignments, internal marks, credits
- Semester info, study materials, academic calendar
- Laboratories, projects, degree requirements, admissions

2. fee
Use for:
- Tuition fees, fee structure, hostel fees, transport fees
- Scholarships, fee payment, payment deadline, refund policy
- Late fee, financial aid, receipts, payment methods
- Education loans, installments

3. general
Use for:
- Hostel, library, sports, clubs, placements, events
- Contact info, office timings, college location, parking
- Wi-Fi, cafeteria, holidays, greetings, or anything else

Rules:
- Choose exactly ONE category.
- Do not explain. Do not answer the question.
- Return ONLY one word: academic / fee / general

User Query:
{user_query}
"""

    result = model.invoke(prompt)
    category = result.content.strip().lower()

 
    if category not in ["academic", "fee", "general"]:
        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"

    return {"query_type": category}

def academic_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    if academic_retriever:
        docs = academic_retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
    else:
        context = "NO_RETRIEVAL_NEEDED"
    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    if fees_retriever:
        docs = fees_retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
    else:
        context = "NO_RETRIEVAL_NEEDED"
    return {"retrieved_context": context}

def general_node(state: State) -> dict:
    """
    No PDF for general questions (hostel, library, sports, events,
    placements etc). We do a quick Tavily web search so the answer
    can include current/real info instead of the LLM just guessing.
    If the search fails for some reason, we fall back to the LLM's
    own knowledge instead of crashing the whole graph.
    """
    query = state["messages"][-1].content

    try:
        search_results = search_tool.invoke({"query": query})
       
        result_items = search_results.get("results", [])
        context = "\n\n".join(
            f"{item.get('title', '')}\n{item.get('content', '')}"
            for item in result_items
        )
        if not context.strip():
            context = "NO_RETRIEVAL_NEEDED"
    except Exception as e:
        print(f"[warning] Tavily search failed: {e}")
        context = "NO_RETRIEVAL_NEEDED"

    return {"retrieved_context": context}

def response_node(state: State) -> dict:
    """Generates the final answer, supportive in friend mode, factual otherwise."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]
    friend_mode = state.get("friend_mode", False)

    if friend_mode:
        # Friend mode → supportive, motivating friend
        prompt = (
            f"You are a supportive, motivating friend for a {programme} student. "
            f"Respond warmly, with encouragement and positivity. "
            f"Keep the tone casual and uplifting, like a friend cheering them on. "
            f"Use emojis naturally to make the conversation feel friendly. "
            f"User’s message: {query}\n\n"
            f"Give a short, clear, and kind response that makes the student feel supported."
        )
    elif context == "NO_RETRIEVAL_NEEDED":
        # General mode without friend toggle → helpful assistant
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        # Academic / Fee mode → factual but approachable
        prompt = (
            f"You are a helpful college assistant talking to a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, precise answer but keep the tone approachable and encouraging."
        )

    response = model.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}