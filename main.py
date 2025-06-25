



import streamlit as st
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal
import time


try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt.tool_node import ToolNode
    from langchain_tavily import TavilySearch
    from langchain.tools import Tool
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    from typing_extensions import TypedDict
    from typing import Annotated
except ImportError as e:
    st.error(f"Missing required packages: {e}")
    st.stop()


    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("python-dotenv not installed. Make sure to set environment variables manually.")


st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .travel-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #10b981;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
        border-left: 4px solid #03a9f4;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Data Models
class TravelPreferences(BaseModel):
    destination: str = Field(description="Travel destination city/country")
    days: int = Field(description="Number of days for the trip")
    interests: List[str] = Field(description="List of interests like culture, food, adventure, etc.")
    budget: int = Field(description="Budget in your local currency")
    travel_type: str = Field(default="leisure", description="Type of travel: leisure, business, adventure")

class State(TypedDict):
    messages: Annotated[List, add_messages]
    preferences: Dict
    current_plan: str
    needs_tools: bool

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'agent': None,
        'chat_history': [],
        'current_preferences': {},
        'travel_plans': [],
        'agent_state': {
            "messages": [],
            "preferences": {},
            "current_plan": "",
            "needs_tools": False
        },
        'show_examples': False,
        'initialized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Agent setup functions
def setup_llm():
   
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY not found in environment variables!")
        st.info("Please set your GROQ API key in the environment variables or .env file")
        return None
    
    try:
        return ChatGroq(
            model="llama3-70b-8192",
            groq_api_key=groq_api_key,
            temperature=0.7
        )
    except Exception as e:
        st.error(f"Failed to initialize GROQ LLM: {e}")
        return None

# Weather API Integration
def get_weather_info(location: str) -> str:
    """Get real weather information using OpenWeatherMap API"""
    weather_api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not weather_api_key:
        return f"Weather API key not configured. Please set WEATHER_API_KEY environment variable. For {location}: Please check local weather forecast before traveling."
    
    try:
        # Current weather
        current_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
        current_response = requests.get(current_url, timeout=10)
        
        if current_response.status_code == 200:
            current_data = current_response.json()
            
            # 5-day forecast
            forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={weather_api_key}&units=metric"
            forecast_response = requests.get(forecast_url, timeout=10)
            
            weather_info = f"🌤️ Weather in {location}:\n"
            weather_info += f"Current: {current_data['weather'][0]['description'].title()}, "
            weather_info += f"{current_data['main']['temp']}°C (feels like {current_data['main']['feels_like']}°C)\n"
            weather_info += f"Humidity: {current_data['main']['humidity']}%, "
            weather_info += f"Wind: {current_data['wind']['speed']} m/s\n"
            
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                weather_info += "\n📅 5-Day Forecast:\n"
                
                # Group forecast by day
                daily_forecasts = {}
                for item in forecast_data['list'][:15]:  # Next 5 days (3-hour intervals)
                    date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
                    if date not in daily_forecasts:
                        daily_forecasts[date] = {
                            'temp_min': item['main']['temp_min'],
                            'temp_max': item['main']['temp_max'],
                            'description': item['weather'][0]['description'],
                            'humidity': item['main']['humidity']
                        }
                    else:
                        daily_forecasts[date]['temp_min'] = min(daily_forecasts[date]['temp_min'], item['main']['temp_min'])
                        daily_forecasts[date]['temp_max'] = max(daily_forecasts[date]['temp_max'], item['main']['temp_max'])
                
                for date, forecast in list(daily_forecasts.items())[:5]:
                    day_name = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
                    weather_info += f"{day_name}: {forecast['description'].title()}, "
                    weather_info += f"{forecast['temp_min']:.1f}°C - {forecast['temp_max']:.1f}°C\n"
            
            weather_info += "\n💡 Travel Tips based on weather:\n"
            temp = current_data['main']['temp']
            if temp < 10:
                weather_info += "• Pack warm clothes, jackets, and layers\n"
            elif temp > 30:
                weather_info += "• Pack light, breathable clothes and sun protection\n"
            else:
                weather_info += "• Pack comfortable clothes for mild weather\n"
                
            if current_data['main']['humidity'] > 70:
                weather_info += "• High humidity - pack moisture-wicking fabrics\n"
                
            return weather_info
        else:
            return f"Could not fetch weather data for {location}. Status code: {current_response.status_code}"
            
    except requests.RequestException as e:
        return f"Weather service temporarily unavailable for {location}. Please check local forecasts. Error: {str(e)}"
    except Exception as e:
        return f"Error fetching weather for {location}: {str(e)}"

# Transportation API Integration
def get_transportation_info(origin: str, destination: str = None) -> str:
    """Get transportation information"""
    
    # If destination is not provided, give general transport info for the origin
    if not destination:
        location = origin
        transport_info = f"🚗 Transportation Options in {location}:\n\n"
        
        # General transportation recommendations based on common destinations
        common_transports = {
            "paris": {
                "public": "Metro, RER trains, buses - €1.90 per trip, day pass €7.50",
                "taxi": "Taxis available, Uber/Bolt operational",
                "bike": "Vélib' bike sharing system available",
                "airport": "CDG/Orly airports - RER B/Orlyval connections"
            },
            "tokyo": {
                "public": "JR Yamanote Line, Tokyo Metro - ¥140-320 per trip",
                "taxi": "Taxis expensive, use for short distances",
                "bike": "Bike rentals available in many areas",
                "airport": "Narita/Haneda - Express trains available"
            },
            "london": {
                "public": "London Underground, buses - £2.50-5.50 per trip",
                "taxi": "Black cabs, Uber available",
                "bike": "Boris Bikes (Santander Cycles)",
                "airport": "Heathrow/Gatwick/Stansted - Express trains"
            },
            "new york": {
                "public": "NYC Subway, buses - $2.90 per trip",
                "taxi": "Yellow cabs, Uber/Lyft widely available",
                "bike": "Citi Bike sharing system",
                "airport": "JFK/LGA/EWR - AirTrain and express buses"
            },
            "mumbai": {
                "public": "Local trains, BEST buses, Metro - ₹10-50 per trip",
                "taxi": "Auto-rickshaws, Ola/Uber available",
                "bike": "Bike rentals limited, scooter rentals available",
                "airport": "Mumbai Airport - Metro and taxi connections"
            },
            "delhi": {
                "public": "Delhi Metro, DTC buses - ₹10-60 per trip",
                "taxi": "Auto-rickshaws, Ola/Uber widely available",
                "bike": "Bike rentals available in some areas",
                "airport": "IGI Airport - Metro and express buses"
            }
        }
        
        location_lower = location.lower()
        transport_data = None
        
        for city, data in common_transports.items():
            if city in location_lower or location_lower in city:
                transport_data = data
                break
        
        if transport_data:
            transport_info += f"🚇 Public Transport: {transport_data['public']}\n"
            transport_info += f"🚕 Taxi/Rideshare: {transport_data['taxi']}\n"
            transport_info += f"🚲 Bike Sharing: {transport_data['bike']}\n"
            transport_info += f"✈️ Airport Access: {transport_data['airport']}\n"
        else:
            transport_info += "🚇 Public Transport: Check local metro/bus systems\n"
            transport_info += "🚕 Taxi/Rideshare: Local taxis and ride-sharing apps\n"
            transport_info += "🚲 Bike Sharing: Look for local bike rental services\n"
            transport_info += "✈️ Airport Access: Airport express services usually available\n"
        
        transport_info += "\n💡 General Transportation Tips:\n"
        transport_info += "• Download local transport apps for real-time info\n"
        transport_info += "• Consider day/week passes for multiple trips\n"
        transport_info += "• Keep cash handy for some local transport\n"
        transport_info += "• Check transport operating hours\n"
        
        return transport_info
    
    # For route planning between cities/countries (simplified)
    transport_info = f"🗺️ Transportation from {origin} to {destination}:\n\n"
    
    # Distance-based recommendations
    route_key = f"{origin.lower()}-{destination.lower()}"
    common_routes = {
        "mumbai-goa": {
            "flight": "1.5 hour flight, ₹3000-8000",
            "train": "12-hour train journey, ₹500-2000", 
            "bus": "10-12 hour bus, ₹800-1500",
            "car": "8-10 hour drive, tolls ~₹500"
        },
        "delhi-mumbai": {
            "flight": "2 hour flight, ₹4000-12000",
            "train": "16-20 hour train journey, ₹1000-5000",
            "bus": "18-24 hour bus, ₹1200-2500",
            "car": "14-16 hour drive, tolls ~₹1000"
        }
    }
    
    if route_key in common_routes or route_key[::-1] in common_routes:
        route_data = common_routes.get(route_key, common_routes.get(route_key[::-1]))
        transport_info += f"✈️ Flight: {route_data['flight']}\n"
        transport_info += f"🚂 Train: {route_data['train']}\n"
        transport_info += f"🚌 Bus: {route_data['bus']}\n"
        transport_info += f"🚗 Car: {route_data['car']}\n"
    else:
        transport_info += "✈️ Flight: Check flight booking sites for best prices\n"
        transport_info += "🚂 Train: Look for rail connections and sleeper options\n"
        transport_info += "🚌 Bus: Consider overnight buses for longer distances\n"
        transport_info += "🚗 Car Rental: Check local rental agencies and road conditions\n"
    
    transport_info += "\n📱 Recommended Booking Apps:\n"
    transport_info += "• Flights: Skyscanner, Kayak, airline direct bookings\n"
    transport_info += "• Trains: Local railway booking websites\n"
    transport_info += "• Buses: RedBus, local bus operators\n"
    transport_info += "• Car Rentals: Hertz, Avis, local providers\n"
    
    return transport_info

def setup_tools():
    """Setup tools for the agent"""
    tools = []
    
    # Search tool for real-time information
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            search_tool = TavilySearch(
                api_key=tavily_api_key,
                max_results=3
            )
            tools.append(Tool(
                name="search_tool",
                func=search_tool.run,
                description="Search for current travel information, attractions, hotels, and travel tips"
            ))
        except Exception as e:
            st.warning(f"Tavily search tool setup failed: {e}")
    
    # Real Weather tool
    tools.append(Tool(
        name="weather_tool",
        func=get_weather_info,
        description="Get real-time weather information and forecast for travel destinations"
    ))
    
    # Transportation tool
    def transport_wrapper(query: str) -> str:
        """Wrapper for transportation tool to handle different query formats"""
        # Try to parse the query for origin and destination
        parts = query.lower().split(' to ')
        if len(parts) == 2:
            return get_transportation_info(parts[0].strip(), parts[1].strip())
        else:
            # Single destination query
            return get_transportation_info(query.strip())
    
    tools.append(Tool(
        name="transportation_tool",
        func=transport_wrapper,
        description="Get transportation options within a city or between cities. Use format 'city' for local transport or 'origin to destination' for route planning"
    ))
    
    # Local attractions tool (keeping existing)
    def get_attractions(location: str) -> str:
        """Get popular attractions for a location"""
        attractions_db = {
            "paris": "Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Arc de Triomphe",
            "tokyo": "Tokyo Tower, Senso-ji Temple, Shibuya Crossing, Mount Fuji day trip",
            "london": "Big Ben, Tower of London, British Museum, Buckingham Palace",
            "new york": "Statue of Liberty, Central Park, Times Square, Empire State Building",
            "delhi": "Red Fort, India Gate, Lotus Temple, Qutub Minar",
            "mumbai": "Gateway of India, Marine Drive, Elephanta Caves, Chhatrapati Shivaji Terminus",
            "goa": "Basilica of Bom Jesus, Se Cathedral, Beaches, Spice Plantations",
            "kerala": "Backwaters, Munnar Tea Gardens, Periyar Wildlife Sanctuary"
        }
        location_lower = location.lower()
        for city, attractions in attractions_db.items():
            if city in location_lower or location_lower in city:
                return f"Popular attractions in {location}: {attractions}"
        return f"Popular attractions in {location}: Historical sites, local markets, cultural centers, and scenic viewpoints"
    
    tools.append(Tool(
        name="attractions_tool",
        func=get_attractions,
        description="Get information about popular tourist attractions in a destination"
    ))
    
    return tools

def setup_rag():
    """Setup RAG (Retrieval Augmented Generation) system"""
    try:
        # Initialize embeddings
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Check if vector store exists
        vector_store_path = "vector_store"
        if os.path.exists(os.path.join(vector_store_path, "index.faiss")):
            try:
                faiss_store = FAISS.load_local(
                    vector_store_path, 
                    embedding, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                st.warning(f"Failed to load existing vector store: {e}")
                faiss_store = create_new_vector_store(embedding, vector_store_path)
        else:
            faiss_store = create_new_vector_store(embedding, vector_store_path)
        
        # Create retriever
        retriever = faiss_store.as_retriever(search_kwargs={"k": 3})
        
        # Setup LLM for RAG
        llm = setup_llm()
        if not llm:
            raise Exception("LLM not initialized")
        
        # Create RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        def rag_query(question: str) -> str:
            """Query the RAG system"""
            try:
                result = rag_chain({"query": question})
                return result["result"]
            except Exception as e:
                return f"Unable to retrieve specific information. General travel advice: Plan ahead, check visa requirements, and research local customs."
        
        return Tool(
            name="rag_tool",
            func=rag_query,
            description="Get detailed destination information from travel guides and knowledge base"
        )
    
    except Exception as e:
        st.warning(f"RAG setup encountered an issue: {e}")
        # Return a fallback tool
        return Tool(
            name="rag_tool",
            func=lambda x: "Travel knowledge base temporarily unavailable. Using general travel advice.",
            description="Fallback travel information tool"
        )

def create_new_vector_store(embedding, vector_store_path):
    """Create a new vector store with sample travel data"""
    from langchain.schema import Document
    
    sample_docs = [
        "Paris is known for its iconic Eiffel Tower, world-class museums like the Louvre, and charming cafes along the Champs-Élysées.",
        "Tokyo offers a blend of traditional culture and modern technology, with highlights including temples, sushi restaurants, and bustling districts like Shibuya.",
        "London features historic landmarks like Big Ben and modern attractions, plus excellent museums and parks.",
        "Mumbai is the financial capital of India, famous for Bollywood, street food, and attractions like Gateway of India and Marine Drive.",
        "New York City is famous for its skyline, Broadway shows, Central Park, and diverse neighborhoods.",
        "Goa is known for its beautiful beaches, Portuguese colonial architecture, and vibrant nightlife.",
        "Kerala is famous for its backwaters, hill stations like Munnar, and Ayurvedic treatments."
    ]
    
    documents = [Document(page_content=doc) for doc in sample_docs]
    faiss_store = FAISS.from_documents(documents, embedding)
    
    # Save the vector store
    try:
        os.makedirs(vector_store_path, exist_ok=True)
        faiss_store.save_local(vector_store_path)
    except Exception as e:
        st.warning(f"Failed to save vector store: {e}")
    
    return faiss_store

# Agent nodes
def extract_preferences_node(state: State):
    """Extract travel preferences from user input"""
    llm = setup_llm()
    if not llm:
        return {"messages": [AIMessage(content="Unable to initialize AI model. Please check your API keys.")]}
    
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # Handle greetings
    greetings = ["hi", "hello", "hey", "hii", "start", "help"]
    if any(greeting in last_message.lower() for greeting in greetings) and len(state["messages"]) <= 1:
        welcome_msg = """🌍 Welcome to your Personal Travel Assistant! 

I can help you plan amazing trips! Please tell me:
• Where would you like to go?
• How many days do you want to travel?
• What are your interests? (culture, food, adventure, relaxation, etc.)
• What's your budget?

Example: "I want to visit Paris for 5 days, interested in art and food, budget is $2000"
"""
        return {
            "messages": [AIMessage(content=welcome_msg)],
            "needs_tools": False
        }
    
    try:
        # Try to extract preferences using Pydantic parser
        parser = PydanticOutputParser(pydantic_object=TravelPreferences)
        preference_prompt = PromptTemplate(
            template="""
You are a travel preference extraction expert. Extract travel preferences from the user's message.
If any information is missing, use reasonable defaults or ask for clarification.

User Message: {input}

{format_instructions}

If you cannot extract clear preferences, respond with a question to get more details.
""",
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        extraction_result = preference_prompt.format(input=last_message)
        response = llm.invoke([HumanMessage(content=extraction_result)])
        
        try:
            preferences = parser.parse(response.content)
            return {
                "preferences": preferences.model_dump(),
                "messages": [AIMessage(content=f"Got it! Planning a {preferences.days}-day trip to {preferences.destination} focused on {', '.join(preferences.interests)} with a budget of {preferences.budget}. Let me create a detailed plan for you!")],
                "needs_tools": False
            }
        except Exception:
            # If parsing fails, have a conversation
            conversation_response = llm.invoke([
                SystemMessage(content="You are a helpful travel assistant. If the user hasn't provided complete travel information (destination, days, interests, budget), ask friendly questions to get the missing details."),
                HumanMessage(content=last_message)
            ])
            return {
                "messages": [AIMessage(content=conversation_response.content)],
                "needs_tools": False
            }
    
    except Exception as e:
        return {
            "messages": [AIMessage(content="I'd love to help you plan your trip! Could you tell me where you'd like to go, for how many days, what you're interested in, and your budget range?")],
            "needs_tools": False
        }

def planning_node(state: State):
    """Create travel plan based on preferences"""
    llm = setup_llm()
    if not llm:
        return {"messages": [AIMessage(content="Unable to initialize AI model. Please check your API keys.")]}
    
    if not state.get("preferences"):
        return {"messages": [AIMessage(content="Let me know your travel preferences first!")]}
    
    prefs = state["preferences"]
    destination = prefs.get("destination", "Unknown")
    days = prefs.get("days", 5)
    interests = prefs.get("interests", [])
    budget = prefs.get("budget", 1000)
    
    plan_prompt = f"""
Create a detailed {days}-day travel itinerary for {destination} with the following preferences:
- Interests: {', '.join(interests)}
- Budget: {budget}
- Travel type: {prefs.get('travel_type', 'leisure')}

Include:
1. Daily itinerary with specific activities
2. Estimated costs for major expenses
3. Travel tips and recommendations
4. Best time to visit attractions
5. Local cuisine recommendations
6. Transportation suggestions

Make it practical and exciting!
"""
    
    try:
        plan_response = llm.invoke([
            SystemMessage(content="You are an expert travel planner. Create detailed, practical, and exciting travel itineraries."),
            HumanMessage(content=plan_prompt)
        ])
        
        travel_plan = plan_response.content
        
        return {
            "current_plan": travel_plan,
            "messages": [AIMessage(content=f"🎯 Here's your personalized {days}-day {destination} travel plan:\n\n{travel_plan}")],
            "needs_tools": False
        }
    
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"I can help you plan a wonderful {days}-day trip to {destination}! Let me gather some current information about the destination.")],
            "needs_tools": True
        }

def tool_calling_node(state: State):
    """Handle tool calling for additional information"""
    return {"messages": [AIMessage(content="Let me gather some current information for you...")]}

def router(state: State) -> Literal["planning", "tools", "extract_preferences", "__end__"]:
    """Route the conversation flow"""
    if not state.get("messages"):
        return "extract_preferences"
    
    last_message = state["messages"][-1]
    
    # Prevent infinite loops
    if len(state["messages"]) > 10:
        return "__end__"
    
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        
        if state.get("preferences") and not state.get("current_plan"):
            return "planning"
        elif state.get("preferences"):
            if any(keyword in content for keyword in ["weather", "attractions", "search", "find", "tell me about", "transport", "transportation"]):
                return "tools"
            elif any(keyword in content for keyword in ["thanks", "thank you", "bye", "goodbye"]):
                return "__end__"
            else:
                return "__end__"
        else:
            return "extract_preferences"
    
    elif isinstance(last_message, AIMessage):
        if state.get("preferences") and not state.get("current_plan"):
            return "planning"
        elif state.get("current_plan"):
            return "__end__"
    
    return "__end__"

def create_travel_agent():
    """Create the LangGraph travel agent"""
    try:
        # Setup tools
        tools = setup_tools()
        rag_tool = setup_rag()
        tools.append(rag_tool)
        
        # Create tool node
        tool_node = ToolNode(tools)
        
        # Create workflow
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("extract_preferences", extract_preferences_node)
        workflow.add_node("planning", planning_node)
        workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.set_entry_point("extract_preferences")
        
        # Add conditional edges
        workflow.add_conditional_edges("extract_preferences", router)
        workflow.add_conditional_edges("planning", router)
        workflow.add_conditional_edges("tools", router)
        
        # Compile the workflow
        return workflow.compile()
        
    except Exception as e:
        st.error(f"Failed to create travel agent: {e}")
        return None

# Streamlit UI
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Check environment variables
    if not check_environment():
        return
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ AI Travel Planner</h1>
        <p>Your Personal Travel Assistant - Plan Amazing Trips with AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    render_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_chat_interface()
    
    with col2:
        render_info_panel()

def check_environment():
    """Check if required environment variables are set"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    weather_api_key = os.getenv("WEATHER_API_KEY")
    
    if not groq_api_key:
        st.error("""
        ❌ **Missing Required API Key:**
        
        Please set the GROQ_API_KEY environment variable.
        
        **Setup Instructions:**
        1. Get your API key from https://console.groq.com/
        2. Set GROQ_API_KEY in your environment
        3. Restart the application
        """)
        
        st.info("""
        **Optional API Keys for Enhanced Features:**
        - WEATHER_API_KEY: Get from https://openweathermap.org/api
        - TAVILY_API_KEY: Get from https://tavily.com/
        """)
        return False
    
    return True

def render_sidebar():
    """Render the sidebar with configuration and examples"""
    st.sidebar.title("🎯 Travel Planner")
    
    # API Status
    st.sidebar.subheader("📡 API Status")
    
    groq_status = "✅" if os.getenv("GROQ_API_KEY") else "❌"
    weather_status = "✅" if os.getenv("WEATHER_API_KEY") else "⚠️"
    tavily_status = "✅" if os.getenv("TAVILY_API_KEY") else "⚠️"
    
    st.sidebar.markdown(f"""
    - **GROQ AI**: {groq_status}
    - **Weather API**: {weather_status}
    - **Search API**: {tavily_status}
    """)
    
    # Quick Actions
    st.sidebar.subheader("🚀 Quick Actions")
    
    if st.sidebar.button("🔄 Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.current_preferences = {}
        st.session_state.agent_state = {
            "messages": [],
            "preferences": {},
            "current_plan": "",
            "needs_tools": False
        }
        st.rerun()
    
    if st.sidebar.button("📝 Show Examples"):
        st.session_state.show_examples = not st.session_state.show_examples
    
    # Examples
    if st.session_state.show_examples:
        st.sidebar.subheader("💡 Example Queries")
        examples = [
            "Plan a 7-day trip to Japan, interested in culture and food, budget $3000",
            "I want to visit Paris for 5 days with a budget of €2000",
            "Plan a beach vacation in Goa for 4 days, budget ₹50000",
            "What's the weather like in London?",
            "Show me attractions in New York City",
            "Transportation options in Tokyo"
        ]
        
        for example in examples:
            if st.sidebar.button(f"📌 {example[:30]}...", key=f"example_{hash(example)}"):
                st.session_state.user_input = example
    
    # Travel Tips
    st.sidebar.subheader("🧳 Travel Tips")
    st.sidebar.markdown("""
    **Before You Travel:**
    - Check visa requirements
    - Get travel insurance
    - Notify your bank
    - Check passport validity
    
    **Packing Essentials:**
    - Universal adapter
    - Portable charger
    - Copy of documents
    - First aid kit
    """)

def render_chat_interface():
    """Render the main chat interface"""
    st.subheader("💬 Chat with Your Travel Assistant")
    
    # Initialize agent if not already done
    if not st.session_state.initialized:
        with st.spinner("🔧 Initializing AI Travel Assistant..."):
            st.session_state.agent = create_travel_agent()
            if st.session_state.agent:
                st.session_state.initialized = True
                st.success("✅ Travel Assistant Ready!")
            else:
                st.error("❌ Failed to initialize Travel Assistant")
                return
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑 You:</strong> {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            # AI message
            st.markdown(f"""
            <div class="chat-message ai-message">
                <strong>🤖 Travel Assistant:</strong> {ai_msg}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your travel question or preferences...")
    
    if user_input:
        process_user_input(user_input)

def process_user_input(user_input):
    """Process user input and generate response"""
    if not st.session_state.agent:
        st.error("Travel Assistant not initialized. Please refresh the page.")
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append((user_input, ""))
    
    # Update agent state
    st.session_state.agent_state["messages"].append(HumanMessage(content=user_input))
    
    try:
        with st.spinner("🤔 Thinking..."):
            # Run the agent
            result = st.session_state.agent.invoke(st.session_state.agent_state)
            
            # Extract AI response
            if result.get("messages"):
                ai_response = result["messages"][-1].content
                
                # Update session state
                st.session_state.agent_state = result
                if result.get("preferences"):
                    st.session_state.current_preferences = result["preferences"]
                
                # Update chat history
                st.session_state.chat_history[-1] = (user_input, ai_response)
                
                # Check if we need to call tools
                if result.get("needs_tools", False):
                    handle_tool_calls(result)
            
            else:
                st.session_state.chat_history[-1] = (user_input, "I'm having trouble processing that. Could you try rephrasing your request?")
    
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)[:100]}..."
        st.session_state.chat_history[-1] = (user_input, error_msg)
        st.error(f"Error processing request: {e}")
    
    st.rerun()

def handle_tool_calls(result):
    """Handle tool calls for additional information"""
    try:
        preferences = result.get("preferences", {})
        destination = preferences.get("destination", "")
        
        if destination:
            # Get weather information
            weather_info = get_weather_info(destination)
            
            # Get transportation information
            transport_info = get_transportation_info(destination)
            
            # Combine information
            additional_info = f"\n\n📊 **Additional Information:**\n\n{weather_info}\n\n{transport_info}"
            
            # Update the last AI message
            if st.session_state.chat_history:
                current_response = st.session_state.chat_history[-1][1]
                st.session_state.chat_history[-1] = (
                    st.session_state.chat_history[-1][0],
                    current_response + additional_info
                )
    
    except Exception as e:
        st.warning(f"Could not fetch additional information: {e}")

def render_info_panel():
    """Render the information panel"""
    st.subheader("📋 Trip Information")
    
    # Current preferences
    if st.session_state.current_preferences:
        st.markdown("### 🎯 Current Trip Preferences")
        prefs = st.session_state.current_preferences
        
        st.markdown(f"""
        <div class="travel-card">
            <strong>🏙️ Destination:</strong> {prefs.get('destination', 'Not specified')}<br>
            <strong>📅 Duration:</strong> {prefs.get('days', 'Not specified')} days<br>
            <strong>🎨 Interests:</strong> {', '.join(prefs.get('interests', []))}<br>
            <strong>💰 Budget:</strong> {prefs.get('budget', 'Not specified')}<br>
            <strong>🧳 Travel Type:</strong> {prefs.get('travel_type', 'leisure')}
        </div>
        """, unsafe_allow_html=True)
    
    # Quick tools
    st.markdown("### 🔧 Quick Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌤️ Check Weather", key="weather_btn"):
            if st.session_state.current_preferences.get('destination'):
                destination = st.session_state.current_preferences['destination']
                weather = get_weather_info(destination)
                st.session_state.chat_history.append(("Check weather", weather))
                st.rerun()
            else:
                st.warning("Please specify a destination first!")
    
    with col2:
        if st.button("🚗 Transportation", key="transport_btn"):
            if st.session_state.current_preferences.get('destination'):
                destination = st.session_state.current_preferences['destination']
                transport = get_transportation_info(destination)
                st.session_state.chat_history.append(("Transportation info", transport))
                st.rerun()
            else:
                st.warning("Please specify a destination first!")
    
    # Recent plans
    if st.session_state.travel_plans:
        st.markdown("### 📚 Recent Plans")
        for i, plan in enumerate(st.session_state.travel_plans[-3:]):
            with st.expander(f"Plan {i+1}: {plan.get('destination', 'Unknown')}"):
                st.write(plan.get('summary', 'No summary available'))
    
    # Help section
    st.markdown("### ❓ Need Help?")
    st.markdown("""
    **Getting Started:**
    1. Tell me where you want to go
    2. Specify your trip duration
    3. Share your interests
    4. Mention your budget
    
    **Example:** "I want to visit Tokyo for 7 days, interested in culture and food, budget $2500"
    
    **Available Commands:**
    - Ask about weather
    - Request transportation info
    - Get attraction recommendations
    - Modify existing plans
    """)

# Run the application
if __name__ == "__main__":
    main()