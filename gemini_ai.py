from google import genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file")

# Initialize client
client = genai.Client(api_key=api_key)


def fallback_answer(question):
    return "AI service is not working right now. Please try again later."


def generate_answer(question):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Answer this question clearly and concisely:\n{question}"
        )

        return response.text

    except Exception:
        return fallback_answer(question)
    st.write("Test Change")