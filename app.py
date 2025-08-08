# app.py

import os
import json
from dotenv import load_dotenv

# --- Core LangChain components with updated imports ---
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

# --- Community and integration components ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma  # <-- MODIFIED: Using the new Chroma import
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Pydantic for data modeling ---
from pydantic import BaseModel, Field
from typing import List, Optional

# --- 0. Load Environment Variables ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")

# --- Define the desired JSON output structure (Pydantic model) ---
class AnalysisDetail(BaseModel):
    parameter: str = Field(description="The specific parameter being evaluated (e.g., Waiting Period, Age Limit).")
    status: str = Field(description="The compliance status (e.g., Compliant, Non-Compliant).")
    clause_id: str = Field(description="The specific clause ID from the document this decision is based on.")
    reasoning: str = Field(description="The detailed reasoning for the status, citing the clause.")

class JsonResponse(BaseModel):
    decision: str = Field(description="The final decision, either 'Approved' or 'Rejected'.")
    amount: Optional[int] = Field(description="The payable amount. Null if not applicable.")
    justification: str = Field(description="A summary of the overall justification for the decision.")
    analysis_details: List[AnalysisDetail] = Field(description="A list of evaluations for each relevant policy parameter.")

def setup_rag_pipeline(doc_path: str, persist_directory: str = "./chroma_db_gemini"):
    print("--- Stage 1 & 2: Ingestion, Chunking, and Embedding ---")
    loader = TextLoader(doc_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Use the new Chroma class
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print(f"Created and persisted Gemini vector store at '{persist_directory}'.")
    return vectorstore

def query_system(query: str, persist_directory: str = "./chroma_db_gemini"):
    print("\n--- Starting Query Process with Gemini ---")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Use the new Chroma class to load the store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("\n--- Stage 5: Generating Response (with Gemini) ---")
    
    # --- MODIFIED: Initialize LLM correctly, without the deprecated flag ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    # --- MODIFIED: Create a more robust prompt with distinct System and Human messages ---
    system_prompt_text = """
    You are an expert insurance claims adjudicator. Your task is to evaluate a user's query against a provided insurance policy document.
    You MUST respond ONLY with a valid JSON object that strictly follows this structure: {json_structure}
    Do not include any other text, explanations, or conversational filler before or after the JSON object.
    """
    
    human_prompt_text = """
    **Policy Context (Relevant Clauses):**
    {context}

    **User Query:**
    {question}

    **Instructions:**
    1. Analyze the user's query and the policy context.
    2. Evaluate the query against each relevant clause.
    3. Make a final decision: "Approved" or "Rejected".
    4. Provide clear justification and reference the exact clause IDs.
    """

    parser = JsonOutputParser(pydantic_object=JsonResponse)
    
    # Create the full prompt template from system and human messages
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                system_prompt_text,
                partial_variables={"json_structure": parser.get_format_instructions()}
            ),
            HumanMessagePromptTemplate.from_template(human_prompt_text)
        ]
    )
    
    # The chain remains the same
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )

    response_json = rag_chain.invoke(query)
    return response_json

if __name__ == "__main__":
    DOC_PATH = "policy_document.txt"
    DB_DIR = "./chroma_db_gemini"

    # We might need to recreate the DB if the old one causes issues, but let's try skipping first.
    if not os.path.exists(DB_DIR):
        print(f"Gemini database not found. Running setup...")
        setup_rag_pipeline(DOC_PATH, persist_directory=DB_DIR)
    else:
        print(f"Gemini database found at '{DB_DIR}'. Skipping setup.")

    user_query = "I am a 46-year-old male who had knee surgery in Pune. My insurance policy is only 3 months old. Is my claim covered?"
    final_response = query_system(user_query, persist_directory=DB_DIR)

    print("\n--- Final Structured Response (from Gemini) ---")
    print(json.dumps(final_response, indent=2))