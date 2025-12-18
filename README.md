# Physical AI & Humanoid Robotics - AI DocBook

**Developed by: Ubaid Ur Rehman**

This is an advanced documentation platform featuring an AI-powered RAG (Retrieval-Augmented Generation) Chatbot.

## Features

- **AI Chatbot:** Powered by Gemini 2.5 Flash.
- **Vector Database:** Uses Qdrant Cloud for fast knowledge retrieval.
- **Smart Context:** Highlight any text on the page, and the AI will answer based on that specific context.
- **Backend:** FastAPI with Python.
- **Frontend:** Docusaurus (React).

## Setup

1. Go to `/backend`, install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Set your `.env` with Gemini and Qdrant keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

3. Run the backend:
   ```bash
   python main.py
   ```

4. In a new terminal, run the frontend from the root folder:
   ```bash
   npm start
   ```

## Usage

- Open the documentation site at `http://localhost:3000`
- Click the "Ask AI" button to open the chatbot
- Highlight any text on the page to use it as context for your questions
- Ask questions about Physical AI and Humanoid Robotics
