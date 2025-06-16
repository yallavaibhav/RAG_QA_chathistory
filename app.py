import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

# import uuid
# tempdf = f"./temp_{uuid.uuid4()}.pdf"
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatGroq(groq_api_key = groq_api_key, model = "Llama3-8b-8192")
embeddings=OpenAIEmbeddings()

# Streamlit
st.title("Conversional RAG with PDF uploads and chat histrory")
st.write("Upload Pdf's and chat with their content")

# Input the Groq API key
api_key = st.text_input("Enter your GroqAPI key:", type="password")

# Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    session_id = st.text_input("Session Id", value="default_session")


    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}
    
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        for upload_file in uploaded_files:
            tempdf=f"./temp.pdf"
            with open(tempdf,"wb") as file:
                file.write(upload_file.getvalue())
                file_name=upload_file.name
            loader = PyPDFLoader(tempdf)
            docs = loader.load()
            documents.extend(docs)
    # Split and create embeddings for the documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()


        contextulize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "just reformulate it if needed and otherwise return it as is"
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextulize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question answering tasks."
            "use the following pieces of retrived context to answer"
            "the question. If you don't know the answer, say that you"
            "don't know. use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )


        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ('system',system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)



        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history, input_messages_key="input",history_messages_key="chat_history", output_messages_key="answer"
        )


        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}

                },
            )

            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History: ", session_history.messages)
    else:
        st.warning("Please enter your key")