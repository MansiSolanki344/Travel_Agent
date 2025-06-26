# 🧭 Travel Planner AI Agent - LangGraph Implementation

An advanced AI-powered travel assistant using **LangGraph**, **tool calling**, and **RAG** with a beautiful **Streamlit** interface.

---

## 🌍 Overview

This AI Agent helps users plan detailed, personalized trips based on their:
- Interests
- Budget
- Duration
- Preferred activities (e.g. art, food, culture, adventure)

---

## ✅ Features

- ✅ Preference Extraction (Pydantic)
- ✅ Vector Search (FAISS + HuggingFace embeddings)
- ✅ RAG System for context-aware travel Q&A
- ✅ Tool Calling with LLMs (weather, attractions, transport, etc.)
- ✅ Day-by-day itinerary generator
- ✅ Streamlit frontend with chat interface

---

## 📁 Folder Structure
Travel Planner AI Agent/

├── .env.example # Example .env file

├── main.py # Streamlit interface

├── requirements.txt

├── rag/

│ ├── books/

│ │ └── Essential India Travel Guide.pdf

│ ├── clean/

│ │ └── Essential India Travel Guide.txt

│ ├── vector_store/

│ │ ├── index.faiss
│ │ └── index.pkl
│ ├── load_and_embed.py # FAISS index creator

│ ├── pdf_to_text.py # Extract text from PDF

│ └── travel_vector_store/

---

## 🛠 Setup Instructions

1. 🔽 Clone the Repository
```bash
git clone https://github.com/MansiSolanki344/Travel_Agent.git
cd Travel_Agent
2. 🐍 Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. 📦 Install Dependencies
pip install -r requirements.txt
4. 🔐 Set Up Environment Variables

GROQ_API_KEY=""
TAVILY_API_KEY="tvly-dev-""
OPENWEATHER_API_KEY=""
OPENAI_API_KEY=""

Workflow & Approach
User Input ➡️ Preference Extraction ➡️ Planner Node ➡️ Tool Calling ➡️ RAG Retrieval ➡️ Itinerary Generation
   +---------------------+
   |   User Input (UI)   |
   +---------------------+
              |
              v
 +---------------------------+
 | Extract Preferences Node  |
 +---------------------------+
              |
              v
 +---------------------------+
 |        Router Node        |
 +---------------------------+
       /        |        \
      v         v         v
+--------+ +-------------+ +-------------+
| Plan   | | Tool Caller | |  RAG Node   |
| Node   | |   (APIs)    | | (Vector DB) |
+--------+ +-------------+ +-------------+
       \         |        /
        \        v       /
         +------------------+
         | Response Builder |
         +------------------+

Step-by-Step Breakdown:
1. User Input (via Streamlit)
Users describe their travel needs in natural language.
Example: "I want to go to Paris for 5 days. I love food and museums. Budget: $2000"

2. Preference Extraction (extract_preferences_node)
Uses a Pydantic model to extract structured data like:
Duration
Interests
Budget
Location

3. Routing Logic (router_node)
Smart routing determines next step based on available preferences and state.

4. Itinerary Planning (planning_node)
Constructs a personalized travel plan.
Uses preferences to generate daily activities.

5. Tool Calling (tool_calling_node)
Integrates external APIs for:
🧭 Attractions (Tavily)
☁️ Weather (OpenWeatherMap)
🚗 Transportation
📖 RAG Knowledge Base

6. Vector Retrieval (RAG)
Loads FAISS vector index to retrieve semantic chunks of travel knowledge.
Uses sentence-transformers to embed and compare.

7. Response Generation
Combines structured data, retrieved knowledge, and tool responses into a natural reply.



⚙️ Run Instructions
1. Convert PDF to Text
Go to the rag/ folder and run:
python pdf_to_text.py

2. Create Vector Store (FAISS Index)
python load_and_embed.py

3. Start the Streamlit App
streamlit run main.py


💡 Example Usage
User:
"I want to visit Goa for 4 days, I love beaches and food, budget is ₹25,000"

AI:
Provides a 4-day detailed itinerary, budget estimation, travel tips, and real-time weather info.

🔧 Tools Used
🔍 FAISS Vector Search

🧠 HuggingFace Sentence Transformers

☁️ OpenWeatherMap API

🧭 Tavily API

🌐 LangGraph / LangChain for workflow & tool calling

🖥 Streamlit frontend



