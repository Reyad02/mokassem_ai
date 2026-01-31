from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import dotenv_values

env_vars = dotenv_values(".env")
GEMINI_API_KEY = env_vars.get("GEMINI_API_KEY")

db = SQLDatabase.from_uri("sqlite:///data/db.sqlite3")

llm  = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=GEMINI_API_KEY)

system_prompt = """
You are an expert SQL assistant.

Rules:
- ALWAYS inspect foreign key relationships.
- NEVER return raw foreign key IDs alone.
- ALWAYS JOIN related tables to return meaningful details.
- If a table has agent_id JOIN the related table.
- Prefer SELECT with JOINs over simple SELECT.
- give all the details available in the tables like description, total_clicks, area_sqft, baths, beds, currency, location, price, about_property, highlights, status and full_name, email , phone.
- Return complete in json format.
"""

sql_agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    system_prompt=system_prompt,
    # prompt="""You are an expert SQL assistant. Use the following format:
    
    # User Question: {input}
    # SQLQuery: {sql_query}
    # SQLResult: {sql_result}
    # Final Answer: {final_answer}""",
    
)

print("SQL Chatbot is ready!")
print("Type 'exit' or 'quit' to stop.\n")

# Chat loop
while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print(" Goodbye!")
        break
    
    try:
        user_input = user_input + " Provide full details of property like description, total_clicks, area_sqft, baths, beds, currency, location, price, about_property, highlights, status and agent full_name, agent email , agent phone. Return the response in json format."
        response = sql_agent.invoke(
            {"input": user_input}
        )
        print("\nBot:", response["output"], "\n")

    except Exception as e:
        print("Error:", str(e), "\n")