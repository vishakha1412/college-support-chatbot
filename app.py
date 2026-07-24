import os
import uuid
import streamlit as st
from college_assistant import graph, build_retriever

 
st.set_page_config(page_title="College Saathi 🎓", page_icon="🎓", layout="wide")
 
defaults = {
    "chat_history": [],
    "thread_id": str(uuid.uuid4()),
    "programme": "",
    "friend_mode": False,
    "documents_loaded": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

 
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #e6ffe6 0%, #ccffcc 100%);
    }

    /* Chat bubbles */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="stChatMessage-human"] {
        background-color: #d9f2d9;
        border: 1px solid #66cc66;
    }
    .stChatMessage[data-testid="stChatMessage-ai"] {
        background-color: #f0fff0;
        border: 1px solid #99cc99;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #e6ffe6 0%, #b3ffb3 100%);
        color: #003300;
    }

    /* Buttons */
    .stButton>button {
        background-color: #66cc66;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4da64d;
        color: #fff;
    }

    /* Titles */
    h1, h2, h3 {
        color: #006600;
    }
    </style>
""", unsafe_allow_html=True)


 
st.title("🎓 College Saathi")
st.caption("Your vibrant campus buddy — helpful, entertaining, and smart!")

 
st.subheader("📂 Upload College Documents")
academic_pdf = st.file_uploader("Upload Academic Handbook", type=["pdf"])
fee_pdf = st.file_uploader("Upload Fee Structure", type=["pdf"])

if st.button("🚀 Initialize Chatbot"):
    if academic_pdf is None or fee_pdf is None:
        st.error("Please upload both PDFs.")
        st.stop()

    os.makedirs("uploads", exist_ok=True)
    academic_path = "uploads/academic.pdf"
    fee_path = "uploads/fee.pdf"

    with open(academic_path, "wb") as f:
        f.write(academic_pdf.getbuffer())
    with open(fee_path, "wb") as f:
        f.write(fee_pdf.getbuffer())

    with st.spinner("⚙️ Creating knowledge base..."):
        try:
            build_retriever(academic_path)
            build_retriever(fee_path)
            st.session_state.documents_loaded = True
            st.success("✅ Knowledge base ready!")
        except Exception as e:
            st.error(f"Error creating knowledge base: {e}")

 
with st.sidebar:
    st.header("👤 Student Details")
    programme_input = st.text_input(
        "Which programme are you in?",
        value=st.session_state.programme,
        placeholder="e.g. BCA, BBA, B.Com(H), B.Tech CSE, MBA..."
    )
    st.session_state.programme = programme_input

    st.session_state.friend_mode = st.checkbox(
        "🤝 Friend Mode (Supportive Chat)",
        value=st.session_state.friend_mode
    )

    st.divider()
    if st.button("🔄 Start New Chat"):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.caption("✨ Powered by RAG + Tavily API + Web Search")

 
if not st.session_state.documents_loaded:
    st.info("Upload both PDFs and initialize the chatbot.")
    st.stop()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "human" else "🤖"):
        st.markdown(message["content"])

user_query = st.chat_input("Ask me anything...")
if user_query:
    if not st.session_state.programme.strip():
        st.warning("⚠️ Please enter your programme in the sidebar first.")
        st.stop()

    if st.session_state.friend_mode:
        st.markdown(
            "<div style='background-color:#e6ffe6; padding:10px; border-radius:8px; "
            "border:1px solid #28a745; text-align:center; font-weight:bold;'>"
            "💖 Friend Mode Active — I'm here as your supportive buddy!"
            "</div>",
            unsafe_allow_html=True
        )

    st.session_state.chat_history.append({"role": "human", "content": user_query})
    with st.chat_message("human", avatar="🧑"):
        st.markdown(user_query)

    with st.chat_message("ai", avatar="🤖"):
        with st.spinner("🤔 Thinking..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = graph.invoke(
                    {
                        "programme": st.session_state.programme,
                        "friend_mode": st.session_state.friend_mode,
                        "messages": [("human", user_query)]
                    },
                    config=config
                )
                if "messages" in result and result["messages"]:
                    answer = getattr(result["messages"][-1], "content", str(result["messages"][-1]))
                else:
                    answer = "⚠️ No response from assistant."
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error generating response: {e}")
                answer = f"⚠️ Error: {e}"

    st.session_state.chat_history.append({"role": "ai", "content": answer})
