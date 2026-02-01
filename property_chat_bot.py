from google import  genai   
from dotenv import load_dotenv
load_dotenv()

client = genai.Client()

SYSTEM_PROMPT = """
You are a professional real estate assistant.

Rules:
- You ONLY answer questions related to real estate and properties.
- Topics allowed: buying, selling, renting, property prices, mortgages,
  investment properties, ZIP codes, counties, inspections, ownership,
  land, buildings, and housing markets.
- If a question is NOT related to property or real estate,
  politely refuse and explain your scope in one sentence.
- Keep answers clear, concise, and professional.
"""

# STATIC PROPERTY DATA
PROPERTY_CONTEXT = """
Static Property Knowledge Base:

- Average home price:
  • Urban area: $750,000
  • Suburban area: $420,000

- Common inspection issues:
  • Roof age
  • Plumbing leaks
  • Electrical wiring
  • Foundation cracks

- Good investment properties:
  • Duplex and multi-family units
  • Properties near schools and transit

- Rental tips:
  • Check local rental laws
  • Verify lease terms
  • Inspect before signing
"""

# CHAT LOOP

def run_chatbot():
    print("Property Chatbot is running")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye! ")
            break

        prompt = f"""
{SYSTEM_PROMPT}

{PROPERTY_CONTEXT}

User Question:
{user_input}
"""

        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

            print("\nBot:", response.text, "\n")

        except Exception as e:
            print("Bot: Something went wrong:", str(e))


# ENTRY POINT

if __name__ == "__main__":
    run_chatbot()