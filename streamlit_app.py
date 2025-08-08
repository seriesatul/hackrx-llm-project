# streamlit_app.py (Definitive Final Version)

import os
import json
import streamlit as st
import tempfile
from dotenv import load_dotenv
import asyncio

# --- All imports are the same ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Load Env Vars and Pydantic Models (No change) ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()
class AnalysisDetail(BaseModel):
    parameter: str = Field(description="The specific parameter being evaluated (e.g., Waiting Period, Age Limit).")
    status: str = Field(description="The compliance status (e.g., Compliant, Non-Compliant).")
    clause_id: str = Field(description="The specific clause ID or section from the document this decision is based on.")
    reasoning: str = Field(description="The detailed reasoning for the status, citing the clause.")
class JsonResponse(BaseModel):
    decision: str = Field(description="The final decision, e.g., 'Allowed', 'Not Allowed', 'Information Provided'.")
    justification: str = Field(description="A summary of the overall justification for the decision.")
    analysis_details: List[AnalysisDetail] = Field(description="A list of evaluations for each relevant policy parameter.")

# --- Backend Function (This is where the final prompt fix is) ---
@st.cache_resource(show_spinner="Indexing document...")
def setup_conversational_rag_chain(_uploaded_file):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_uploaded_file.getvalue())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
    
    # Chain 1: History-Aware Retriever (This part is correct and remains the same)
    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    # --- THIS IS THE FIX ---
    # Chain 2: Final Answer Generation (A new, more robust prompt structure)
    # The system prompt sets the persona, and the human prompt contains the data and the final instruction.
    # This makes the JSON instruction the last thing the LLM sees.

    qa_system_prompt = "You are a helpful and expert document analysis assistant. Answer the user's question based on the provided context."
    
    qa_human_prompt_template = """Based on the following context, please answer the user's question.

Context:
{context}

Question:
{input}

Your final output MUST be a valid JSON object that strictly follows this JSON schema: {json_structure}"""
    
    parser = JsonOutputParser(pydantic_object=JsonResponse)
    
    # The prompt now correctly includes the chat_history placeholder
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", qa_human_prompt_template),
    ]).partial(json_structure=parser.get_format_instructions())
    # --- END OF FIX ---

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    combine_docs_chain = question_answer_chain | parser
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    return rag_chain

# --- Streamlit UI (This part is now correct and needs no changes) ---
st.set_page_config(page_title="Conversational Document Assistant", layout="wide")
st.title("💬 Conversational Document Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []
with st.sidebar:
    st.header("1. Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file:
    if "rag_chain" not in st.session_state or uploaded_file.name != st.session_state.get("uploaded_file_name"):
        st.session_state.messages = []
        st.session_state.rag_chain = setup_conversational_rag_chain(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.messages.append(AIMessage(content=f"Document '{uploaded_file.name}' is ready. How can I help you?"))
for message in st.session_state.messages:
    with st.chat_message(message.type):
        content_to_display = message.content
        is_json = False
        if isinstance(content_to_display, str):
            try:
                content_dict = json.loads(content_to_display)
                if isinstance(content_dict, dict):
                    content_to_display = content_dict
                    is_json = True
            except (json.JSONDecodeError, TypeError):
                 pass
        if is_json:
            st.info(f'**Decision: {content_to_display.get("decision")}**')
            st.write(content_to_display.get("justification"))
            with st.expander("Show Detailed Analysis"):
                st.json(content_to_display.get("analysis_details"))
        else:
            st.markdown(content_to_display)
if user_query := st.chat_input("Ask a question about your document..."):
    if "rag_chain" not in st.session_state:
        st.warning("Please upload a document first.")
        st.stop()
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("human"):
        st.markdown(user_query)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            try:
                simplified_history = []
                for msg in st.session_state.messages[:-1]:
                    if isinstance(msg, HumanMessage):
                        simplified_history.append(msg)
                    elif isinstance(msg, AIMessage):
                        try:
                            content_dict = json.loads(msg.content)
                            justification = content_dict.get("justification", "I have provided the information.")
                            simplified_history.append(AIMessage(content=justification))
                        except (json.JSONDecodeError, TypeError):
                            simplified_history.append(msg)
                
                response = st.session_state.rag_chain.invoke({
                    "input": user_query,
                    "chat_history": simplified_history
                })
                structured_answer = response.get("answer", {})
                json_string_answer = json.dumps(structured_answer)
                st.session_state.messages.append(AIMessage(content=json_string_answer))
                st.rerun()
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))