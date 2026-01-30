# User Story Agent 🤖

**Turn customer conversations into actionable Agile User Stories.**

Product Demo: https://www.linkedin.com/posts/kushika-sivaprakasam_aiagents-langchain-agenticai-activity-7419446781098225666-qH7j?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC_rLo8BapJZ86-Yyha3FXCv4iUqyjG8YXc

The **User Story Agent** is an AI-powered tool designed for Product Managers and Agile teams. It automatically analyzes customer meeting transcripts, interviews, and feedback to generate high-quality user stories with acceptance criteria. It also features a **RAG (Retrieval-Augmented Generation)** system to chat with your data and seamless integration with **Trello**.

---

## 🏗️ Tech Stack

This project is built using a modern, modular architecture leveraging state-of-the-art AI and web technologies.

### **Frontend Client**
*   **HTML5 & CSS3**: Clean, responsive grid-based layout with a dark mode aesthetic.
*   **Vanilla JavaScript**: Lightweight client-side logic for handling file uploads (Drag & Drop), API communication (Fetch API), and dynamic DOM manipulation.
*   **Bootstrap Icons**: For intuitive visual elements.
*   **Google Fonts (Inter)**: For modern typography.

### **Backend Server**
*   **Python 3.10+**: Core programming language.
*   **FastAPI**: High-performance async web framework for serving API endpoints (`/api/upload-context`, `/api/chat`, etc.).
*   **Uvicorn**: ASGI server implementation for production-grade performance.
*   **Data Processing**:
    *   `pandas`: Parsing CSV and Excel files.
    *   `pypdf`: Extracting text from PDF documents.
    *   `python-docx`: Reading MS Word files.

### **AI & Intelligence**
*   **LangChain**: Orchestrator framework for chaining LLM calls and managing RAG pipelines.
*   **OpenAI GPT-4o**: The underlying Large Language Model for reasoning, generation, and critique.
*   **FAISS (Facebook AI Similarity Search)**: Efficient local vector database for storing and retrieving context embeddings.
*   **OpenAI Embeddings**: for converting text chunks into vector representations (`text-embedding-3-small`).

### **Tools & Integrations**
*   **Trello API**: Direct integration via `py-trello`/REST to create cards in your backlog.
*   **LangSmith**: (Optional) For tracing, monitoring, and debugging LLM workflows.

---

## 🚀 Key Functionalities

### 1. Multi-Format Transcript Analysis
*   **Support for Diverse Inputs**: Upload transcripts in **.txt, .pdf, .docx, .csv, or .xls** formats.
*   **Smart Parsing**: Automatically extracts usage text from structured files.

### 2. Context-Aware RAG (Retrieval-Augmented Generation)
*   **Knowledge Injection**: Upload your company's "Business Documentation" (Roadmap, Policies, PRDs).
*   **Vector Search**: The system chunks and indexes these documents locally using FAISS.
*   **Hybrid Recall**: When reasoning, the agent combines insights from the *current meeting transcript* AND your *historical business context*.

### 3. Interactive "Chat with Data"
*   **Q&A Interface**: A dedicated chat view to ask specific questions.
    *   *"What was the user's main frustration?"*
    *   *"Does this feature request conflict with our Q3 Security Policy?"*
*   **Source Citation**: Answers are grounded in the provided documents.

### 4. Structured User Story Generation
*   **Agile Standard**: Auto-generates stories in the format: *As a [persona], I want [action], so that [benefit].*
*   **Acceptance Criteria**: Automatically drafts detailed checklist items for each story (e.g., "Verify login button exists").
*   **Refinement**: (Coming soon via LangGraph) Automated critique and iteration.

### 5. Trello Integration
*   **One-Click Sync**: Push generated stories directly to a specific Trello list.
*   **Status Tracking**: UI updates to show which items have been synced.

---

## 📦 Setup & Installation

### Prerequisites
*   Python 3.10 or higher.
*   An OpenAI API Key.
*   Trello API Key & Token for syncing.

### 1. Clone & Install
```bash
# Clone the repository
git clone <your-repo-url>
cd user-story-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_key_here
TRELLO_API_KEY=your_trello_key
TRELLO_TOKEN=your_trello_token
TRELLO_LIST_ID=target_list_id
```

### 3. Run the Application
```bash
python3 app.py
```
The server will start at `http://localhost:8000`.

---

## 📖 Usage Guide

### Step 1: Provide Context
1.  Open `http://localhost:8000`.
2.  **Upload Transcript**: Drag & drop your meeting transcript.
3.  **Upload Business Docs**: Add context files (e.g., "Roadmap.pdf") to the "Business Documentation" zone.

### Step 2: Choose Action
*   **Chat with Data**: Interactive Q&A about the files.
*   **Craft User Stories**: Generate backlog items and sync to Trello.

---

## 📂 Project Structure

```
├── app.py                  # Backend: FastAPI App & Endpoints
├── index.html              # Frontend: Single Page Application
├── user_story_agent.py     # AICore: RAG Pipeline & Agent Logic
├── context_docs/           # Storage: Uploaded context files
├── requirements.txt        # Config: Python Dependencies
├── .env                    # Config: API Keys & Secrets
└── User_Story_Agent_Documentation.docx  # Docs: Detailed Arch
```

---
*Built with LangChain by Kushika Sivaprakasam*
