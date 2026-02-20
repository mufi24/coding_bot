import streamlit as st
import ollama

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="CodeGemma Coder Bot", layout="centered")

st.title("💻 CodeGemma 2B - Coder Assistant")

# ---------------------------------
# Initialize Chat History Properly
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------
# Display Chat History
# ---------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------
# Chat Input
# ---------------------------------
user_input = st.chat_input("Ask coding questions...")

if user_input:
    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # System Prompt for Coder Behavior
    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert software engineer. "
            "Provide clean, efficient, well-formatted code. "
            "Always use proper code blocks with language tags. "
            "Explain briefly but focus on code quality."
        )
    }

    # Call CodeGemma
    response = ollama.chat(
        model="codegemma:2b",
        messages=[system_prompt] + st.session_state.messages
    )

    reply = response["message"]["content"]

    # Add assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    with st.chat_message("assistant"):
        st.markdown(reply)