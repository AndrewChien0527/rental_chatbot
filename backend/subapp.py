import streamlit as st
from utils import classify_input, handle_general_question, handle_rental_post

st.set_page_config(page_title="中文租屋聊天機器人", layout="wide")
st.title("🏠 中文租屋聊天機器人")

# 對話歷史初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for msg in st.session_state.messages:
    role, content = msg["role"], msg["content"]
    with st.chat_message(role):
        st.markdown(content)

# 新用戶輸入
query = st.chat_input("請輸入租屋問題或貼文內容：")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 判斷類型：Q1(一般問題)、Q2(FB貼文)、other
    intent = classify_input(query)

    if intent == "Q1":
        answer = handle_general_question(query)
    elif intent == "Q2":
        answer = handle_rental_post(query)
    else:
        answer = "請提供與租屋相關的問題或貼文內容～"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
