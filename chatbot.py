import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback

file_path = "bgf_manual.txt"

with open(file_path, "r", encoding="utf-8") as file:
    bgf_manual = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_text(bgf_manual)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_texts(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())

question = st.text_input("Ask a question:", "", key="question")

if 'last_keypress' not in st.session_state:
    st.session_state.last_keypress = None

if st.button("Submit") or st.session_state.last_keypress == "Enter":
    # When the submit button is pressed, display the input text
    answer = qa.run(question)
    st.write("Response:", answer)
    question = ""
    


# with get_openai_callback() as cb:
#     answer = qa.run(st.session_state.question)
#     print(answer)
#     print(cb)
