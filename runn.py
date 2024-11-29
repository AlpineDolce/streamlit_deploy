import streamlit as st
from langchain_upstage import ChatUpstage
from dotenv import load_dotenv
from llm import get_llm_response

load_dotenv()

llm = ChatUpstage(model='solar-pro')

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title('무엇이든 물어보세요!')

# 기존 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("채팅을 입력해주세요."):
    st.session_state.messages.append({"role" : "user", "content" : user_input})
    with st.chat_message('user'):
        st.write(user_input)

    llm_result  = get_llm_response(user_input)
    st.session_state.messages.append({"role" : "assistant", "content" : llm_result})
    with st.chat_message('assistant'):
        st.write_stream(llm_result)