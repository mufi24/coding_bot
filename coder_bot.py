import streamlit as st
import ollama

# =========================
# CONFIGURATION
# =========================
DEFAULT_MODEL = "codellama:7b"
MAX_HISTORY = 10  # Max user+assistant turn pairs to send

LANG_CONFIG = {
    "Python": {"emoji": "🐍", "id": "python"},
    "JavaScript": {"emoji": "📜", "id": "javascript"},
    "C++": {"emoji": "🖥️", "id": "cpp"},
    "Java": {"emoji": "☕", "id": "java"},
}

BAD_TOKENS = [
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|file_separator|>",
]

# =========================
# HELPER FUNCTIONS
# =========================
def sanitize_output(text: str) -> str:
    """Remove unwanted internal tokens."""
    for token in BAD_TOKENS:
        text = text.replace(token, "")
    return text


def build_system_prompt(language: str, level: str) -> str:
    """Generate dynamic system prompt."""
    return (
        f"You are PolyMentor AI, a coding tutor specializing in {language}.\n"
        f"The user's skill level is: {level}.\n\n"
        "Rules:\n"
        "- Answer ONLY the user's question. Do not go off-topic.\n"
        "- Always produce correct, runnable code.\n"
        "- Use markdown code blocks with the correct language tag.\n"
        "- Add inline comments for beginners.\n"
        "- Be concise and structured.\n"
        "- Never output internal model tokens.\n\n"
        "Response format:\n"
        "📘 Concept — brief explanation\n"
        "💡 Example — working code\n"
        "🧠 Tip — one useful takeaway\n"
    )


def get_trimmed_history(messages: list, max_pairs: int) -> list:
    """
    Return only the last `max_pairs` user+assistant exchanges.
    This avoids context overflow and stale context confusion.
    """
    # Keep only role: user / assistant messages (not system)
    chat_messages = [m for m in messages if m["role"] in ("user", "assistant")]
    # Take last max_pairs * 2 messages (each pair = 1 user + 1 assistant)
    return chat_messages[-(max_pairs * 2):]


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Mentor AI",
    page_icon="🤖",
    layout="centered"
)

# =========================
# SESSION INIT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    selected_lang = st.selectbox("Select Language", list(LANG_CONFIG.keys()))
    lang_info = LANG_CONFIG[selected_lang]

    level = st.select_slider(
        "User Skill Level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3)
    # ^^^ Lowered default to 0.3 — higher temps cause more hallucinations

    st.divider()
    st.info(
        "💡 **Tip:** If answers seem wrong, try lowering the Temperature slider. "
        "For complex questions, consider a larger model like `llama3` or `codellama`."
    )
    st.divider()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.success(f"Running **{selected_lang}** via Local Ollama")

# =========================
# MAIN UI
# =========================
st.title(f"{lang_info['emoji']} {selected_lang} Mentor AI")
st.caption(f"Skill Level: **{level}** | Model: `{st.session_state.model}`")

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# CHAT INPUT
# =========================
if prompt := st.chat_input(f"Ask a {selected_lang} question..."):

    # 1. Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Save user message AFTER displaying (so it's not in history yet when we trim)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Build context: system prompt + trimmed prior history + current user message
    #    We use trimmed history from BEFORE this message was appended
    prior_history = get_trimmed_history(st.session_state.messages[:-1], MAX_HISTORY)

    context_messages = (
        [{"role": "system", "content": build_system_prompt(selected_lang, level)}]
        + prior_history
        + [{"role": "user", "content": prompt}]
    )

    # 4. Stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            stream = ollama.chat(
                model=st.session_state.model,
                messages=context_messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_ctx": 4096,
                    "num_predict": 1024,
                }
                # ^^^ Removed custom stop tokens — they were cutting output too early
            )

            for chunk in stream:
                # Safely extract content — handle all possible chunk shapes
                try:
                    content = ""
                    if isinstance(chunk, dict):
                        content = chunk.get("message", {}).get("content", "")
                    elif hasattr(chunk, "message"):
                        content = chunk.message.content or ""

                    if content:
                        token = sanitize_output(content)
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                except Exception:
                    continue  # Skip malformed chunks silently

            # Final render — strip any trailing cursor artifact
            full_response = sanitize_output(full_response).strip()

            if full_response:
                response_placeholder.markdown(full_response)
                # 5. Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
            else:
                # Model returned nothing — show a helpful message
                response_placeholder.warning(
                    "⚠️ The model returned an empty response. "
                    "This usually means the model is too small for this prompt. "
                    "Try rephrasing, or switch to a larger model like `codellama:7b`."
                )

        except Exception as e:
            st.error(f"❌ Ollama Error: {str(e)}")
            st.info(
                "Make sure Ollama is running locally and the model is pulled.\n\n"
                "Run: `ollama pull hf.co/MaziyarPanahi/codegemma-2b-GGUF:Q4_K_M`"
            )
