#  Conversational Document Analysis Agent for HackRX

This project is an advanced AI-powered conversational agent designed to analyze, understand, and answer questions about complex, unstructured documents like legal contracts, insurance policies, and compliance manuals. Built for the HackRX hackathon, this application leverages a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware, and auditable answers with conversational memory.

---

### 🚀 Live Demo

**[➡️ Click here to access the live application](https://your-streamlit-app-url.streamlit.app/)**

<!-- 
**ACTION REQUIRED:** Deploy your app to Streamlit Community Cloud and replace the URL above with your public app URL. 
-->

### 🎥 Demo Video

![Application Demo GIF](https://raw.githubusercontent.com/your-username/your-repo-name/main/assets/demo.gif)

<!-- 
**ACTION REQUIRED:** Record a short screen capture of your app in action (uploading a PDF, asking a few follow-up questions). 
1. Create a folder named `assets` in your project.
2. Save the GIF as `demo.gif` inside the `assets` folder.
3. Push the `assets` folder to GitHub.
4. Replace `your-username/your-repo-name` in the URL above with your actual GitHub username and repository name.
(You can use tools like Giphy Capture, ScreenToGif, or Kap to create a GIF).
-->

---

## ✨ Key Features

*   **💬 Conversational Chat Interface:** Engage in a natural, multi-turn conversation with your documents. The AI remembers the context of previous questions to answer follow-ups intelligently.
*   **📄 PDF Document Support:** Upload any text-based PDF, and the system will instantly ingest, process, and index its content for analysis.
*   **🧠 Retrieval-Augmented Generation (RAG):** The core of the system. It finds the most relevant information from the document before generating an answer, ensuring responses are grounded in facts and not hallucinations.
*   **🔍 Structured JSON Output:** The AI provides its analysis in a structured JSON format, detailing its final decision, a human-readable justification, and a breakdown of how each relevant clause supports the conclusion.
*   **☁️ Cloud Deployed:** Fully deployed and publicly accessible via Streamlit Community Cloud.

---

## 🏛️ Project Architecture

This project is built on a sophisticated, multi-stage RAG pipeline that ensures accuracy and traceability.

1.  **Document Ingestion & Chunking:** The user uploads a PDF. The system extracts the text and splits it into smaller, semantically meaningful chunks.
2.  **Embedding & Indexing:** Each text chunk is converted into a numerical vector (an embedding) using Google's `embedding-001` model. These vectors are stored in a `ChromaDB` vector store for efficient similarity searching.
3.  **History-Aware Retrieval:** When a user asks a follow-up question, a **History-Aware Retriever** first rephrases the conversational query into a standalone question based on the chat history. This new question is then used to retrieve the most relevant text chunks from the vector store.
4.  **Answer Generation:** The rephrased question, chat history, and the retrieved document chunks are passed to a powerful LLM (**Google Gemini 1.5 Pro**). A carefully engineered prompt instructs the LLM to act as an expert analyst and generate a structured JSON response.
5.  **UI & State Management:** The Streamlit frontend manages the chat history, displays the structured output in a user-friendly format, and handles all user interactions.

---

## 🛠️ Technology Stack

*   **Core AI/ML:** LangChain, Google Gemini 1.5 Pro, Google Generative AI Embeddings
*   **Web Framework:** Streamlit
*   **Vector Database:** ChromaDB
*   **Document Loading:** PyPDFLoader
*   **Deployment:** Streamlit Community Cloud, GitHub
*   **Language:** Python 3.11+

---

## ⚙️ Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the root directory and add your Google API key:
    ```
    GOOGLE_API_KEY="AIzaSy...YourSecretGoogleApiKeyGoesHere"
    ```

5.  **Run the Application:**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 💡 Challenges & Learnings

This project involved overcoming several real-world engineering challenges:

*   **Challenge:** Deploying `ChromaDB` on Streamlit Cloud resulted in a `RuntimeError` due to an outdated system `sqlite3` version.
    *   **Solution:** Implemented the "pysqlite3-binary patch" by adding the library to `requirements.txt` and adding a special import block at the top of the script to force Python to use the newer, compatible version.

*   **Challenge:** The app crashed with an `asyncio` event loop error during document indexing on Streamlit.
    *   **Solution:** Identified that a dependency required an active event loop. Manually created and set a new `asyncio` event loop at the start of the indexing function to resolve the conflict between synchronous Streamlit and asynchronous libraries.

*   **Challenge:** In multi-turn conversations, the LLM would occasionally fail to produce valid JSON, causing an `OutputParserException`.
    *   **Solution:** Re-engineered the final prompt to be more direct and forceful about the JSON output requirement. Additionally, implemented logic to pass a "simplified" version of the chat history to the LLM (using only the `justification` field from previous AI answers) to prevent confusion from nested JSON in the conversation context.

---

## 🚀 Future Enhancements

*   **Show Source Chunks:** Add a feature to display the exact text chunks retrieved from the document that were used to generate the answer, providing full transparency.
*   **Multi-Document Comparison:** Evolve the system into an agent that can ingest two documents (e.g., a new contract and a standard template) and autonomously compare them to flag differences and risks.
*   **OCR Support:** Integrate OCR libraries like `pytesseract` to enable analysis of scanned PDFs and images.

---

Built with ❤️ for **HackRX**.
