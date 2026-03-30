---
description: how to run the RAG Stock Market system
---

Follow these steps to launch the system and see the results.

### 1. Environment Preparation
Ensure your `.env` file is ready with the required API keys.
// turbo
1. Create a virtual environment and activate it:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
2. Install the necessary dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Optionally verify which Gemini models your API key can access:
   ```powershell
   python -m src.list_models
   ```

### 2. Data Ingestion
This step builds your local **ChromaDB** and **BM25** indices.
// turbo
4. Run the ingestion script:
   ```powershell
   python -m scripts.ingest
   ```
   *Expect: A `data/` folder to be created containing `chroma_db` and `bm25_index.pkl`.*

### 3. Launching the Backend API
Start the FastAPI server that powers the LangGraph agent.
// turbo-all
5. In a **new** terminal (keep venv active):
   ```powershell
   python -m uvicorn api.main:app --reload
   ```
   *Expect: API running at `http://localhost:8000`.*

### 4. Launching the Frontend UI
Start the Streamlit interface to interact with the RAG system.
// turbo-all
6. In another **new** terminal (keep venv active):
   ```powershell
   python -m streamlit run ui/app.py
   ```
   *Expect: Your browser should open to `http://localhost:8501`. If not, click the link in the terminal.*

### 5. Running Evaluations (Optional)
Evaluate the performance using the Ragas metrics.
// turbo
7. Run the evaluation script:
   ```powershell
   python -m src.evaluation
   ```
   *Expect: A `data/eval_results.json` file with performance scores.*
