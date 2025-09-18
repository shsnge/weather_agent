import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import requests  # for API calls

# --- Setup LLM ---
llm = ChatGoogleGenerativeAI(
    api_key="AIzaSyB8nJcvwcf1ikM6aF5yOG8aQ4rLWmdrrGM",  # ⚠️ ideally use st.secrets
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# --- Tool: Weather Fetcher ---
def get_weather(city: str) -> str:
    """Fetch real weather data for a given city from API."""
    url = f"https://p2pclouds.up.railway.app/v1/learn/weather?city={city}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get("weather", f"Weather data: {data}")
        else:
            return str(data)
    except Exception as e:
        return f"Error fetching weather: {e}"

# --- Create agent ---
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# --- Streamlit UI ---
st.title("🌤️ Weather Finder Agent ")

city = st.text_input("Enter a city name:")

if st.button("Get Weather"):
    if city.strip():
        with st.spinner("Fetching weather..."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": f"what is the weather in {city}"}]}
            )
            last_message = response["messages"][-1]
            st.success(f"Weather in {city}: {last_message.content}")
    else:
        st.warning("Please enter a city name first.")
