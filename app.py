import streamlit as st
import html
import backend
from typing import List, Dict

st.set_page_config(page_title="UE.ai — Local RAG Chatbot", layout="wide")

# --- session-state defaults (add this early in your frontend) ---
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "active_chat" not in st.session_state:
    # create default chat
    chat_id = "chat_1"
    st.session_state.sessions[chat_id] = {"title": "Chat 1", "messages": []}
    st.session_state.active_chat = chat_id

# UI / model settings defaults
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "synthesize" not in st.session_state:
    st.session_state.synthesize = True
if "openrouter_api_key" not in st.session_state:
    # keep empty by default; backend will use env var if available
    st.session_state.openrouter_api_key = ""


# ---- Sidebar: chat sessions, upload, settings ----
with st.sidebar:
    st.title("UE.ai")
    st.markdown("---")
    st.subheader("Upload Documents")
    uploaded = st.file_uploader("Upload documents (pdf, txt, docx, pptx)", accept_multiple_files=True, type=["pdf","txt","docx","pptx"])
    if uploaded and st.button("Upload"):
        results = []
        with st.spinner("Uploading..."):
            for f in uploaded:
                try:
                    r = backend.save_and_index_file(f)
                    results.append({"file": f.name, "status": "ok", **r})
                except Exception as e:
                    results.append({"file": f.name, "status": "error", "error": str(e)})
        st.write(results)


# ---- Main UI ----
st.title("UE.ai — Local RAG Chatbot")

# Load chat history reference
active = st.session_state.active_chat
if active not in st.session_state.sessions:
    st.session_state.sessions[active] = {"title": active, "messages": []}
chat_history: List[Dict] = st.session_state.sessions[active]["messages"]

# Chat display (safe HTML, content escaped)
st.markdown(
    '<div style="max-height:65vh; overflow:auto; padding:8px; border-radius:8px; background:rgba(0,0,0,0.03)">',
    unsafe_allow_html=True,
)
for msg in chat_history:
    role = msg.get("role")
    text = msg.get("content", "")
    safe_text = html.escape(text)
    if role == "user":
        st.markdown(
            f"<div style='text-align:right;padding:8px;margin:6px;background:linear-gradient(90deg,#0078ff,#00c6ff);color:white;border-radius:10px;display:inline-block;max-width:80%'>{safe_text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:left;padding:8px;margin:6px;background:#222;color:#eee;border-radius:10px;display:inline-block;max-width:80%'>{safe_text}</div>",
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)

# Input
user_input = st.chat_input("Ask anything you want")
if user_input:
    # append user message
    chat_history.append({"role": "user", "content": user_input})
    st.session_state.sessions[active]["messages"] = chat_history
    st.rerun()

# After rerun, check if last message is from user and not answered
if chat_history and chat_history[-1]["role"] == "user":
    user_text = chat_history[-1]["content"]
    
    placeholder = st.empty()
    with placeholder.container():
        st.write("_Responding..._")

    bot_reply = backend.get_chatbot_response(
        [{"role": "user", "content": user_text}],
        top_k=int(st.session_state.get("top_k", 5)),
        synthesize=bool(st.session_state.get("synthesize", True)),
    )

    chat_history.append({"role": "assistant", "content": bot_reply})
    st.session_state.sessions[active]["messages"] = chat_history
    placeholder.empty()
    st.rerun()