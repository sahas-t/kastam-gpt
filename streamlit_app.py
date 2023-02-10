import streamlit as st

# create a streamlit app with a title and a max width of 1000px
st.title('Kastam GPT Demo')
st.markdown('---')
user_input = st.text_input("Enter your message:", max_chars = 1000)

if user_input:
    response = "Hello! How can I help you today?"
    st.write("Bot:", response)










