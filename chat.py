from fastapi import FastAPI
from utils.helpers import load_sessions, save_sessions, get_or_create_session
from dotenv import dotenv_values
from pydantic import BaseModel
from google import genai
from google.genai import types
import requests
from typing import List
import json

env_vars = dotenv_values(".env")
GEMINI_API_KEY = env_vars.get("GEMINI_API_KEY")
RAPIDAPI_KEY = env_vars.get("RAPIDAPI_KEY")

GEMINI_MODEL = "gemini-3-pro-preview"

class PropertyDetails(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft: int
    
class Amenities(BaseModel):
    has_pool: bool
    has_garage: bool
    has_garden: bool
    view_of_water: bool
    has_balcony: bool
    has_spa: bool
    
class ResponseFormat(BaseModel):
    price: str
    name: str
    property_details: List[PropertyDetails]
    amenities: List[Amenities]
    description: str
    about: str
    image_urls: List[str]
    
# def fetch_uae_properties(
#     purpose: str,
#     property_type: str,
#     completion_status: str
# ):
#     url = "https://uae-real-estate-data-real-time-api.p.rapidapi.com/listings/search"

#     headers = {
#         "x-rapidapi-key": RAPIDAPI_KEY,
#         "x-rapidapi-host": "uae-real-estate-data-real-time-api.p.rapidapi.com"
#     }

#     params = {
#         "purpose": purpose,
#         "property_type": property_type,
#         "completion_status": completion_status
#     }

#     response = requests.get(url, headers=headers, params=params, timeout=10)
#     response.raise_for_status()
#     return response.json()

# uae_property_schema = {
#     "name": "fetch_uae_properties",
#     "description": "Fetch UAE real estate listings using purpose, property type, and completion status",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "purpose": {
#                 "type": "string",
#                 "description": "for-rent or for-sale"
#             },
#             "property_type": {
#                 "type": "string",
#                 "description": "apartments, villas, townhouses"
#             },
#             "completion_status": {
#                 "type": "string",
#                 "description": "completed or off-plan"
#             }
#         },
#         "required": ["purpose", "property_type", "completion_status"]
#     }
# }

# uae_property_tool = types.Tool(
#     function_declarations=[uae_property_schema]
# )

def loopnet_search_by_city(city: str):
    # Step 1: Find city
    city_url = "https://loopnet-api.p.rapidapi.com/loopnet/helper/findCity"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "loopnet-api.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    city_payload = {"keywords": city}

    city_res = requests.post(
        city_url, json=city_payload, headers=headers, timeout=10
    )
    city_res.raise_for_status()
    city_data = city_res.json()

    if not city_data.get("data"):
        return {"error": "City not found"}

    location_id = city_data["data"][0]["id"]

    # Step 2: Property search
    property_url = "https://loopnet-api.p.rapidapi.com/loopnet/sale/advanceSearch"

    property_payload = {
        "locationId": location_id,
        "locationType": "city",
        "page": 1,
        "size": 20,
        "auctions": False
    }

    property_res = requests.post(
        property_url, json=property_payload, headers=headers, timeout=10
    )
    property_res.raise_for_status()

    return {
        "city": city_data["data"][0],
        "properties": property_res.json()
    }

loopnet_search_schema = {
    "name": "loopnet_search_by_city",
    "description": "Find properties by city name. Automatically resolves city ID and fetches listings.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. Los Angeles"
            }
        },
        "required": ["city"]
    }
}

loopnet_tool = types.Tool(
    function_declarations=[loopnet_search_schema]
)

class ResearchChat(BaseModel):
    session_id: str | None = None
    message: str
    type: str 

app = FastAPI()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

@app.post("/chat")
async def chat_endpoint(request: ResearchChat):
    sessions = load_sessions()
    session_id = get_or_create_session(sessions, request.session_id)
    ai_message = ""
    google_search_response_message = {}
    loopnet_data = []
   
    conversation_text = ""
    for m in sessions[session_id]:
        conversation_text += f"User: {m['user_message']}\nAI: {m['ai_message']}\n"
    conversation_text += f"User: {request.message}\nAI:"
    
    if request.type == "research":
        system_prompt = f"""
        You are a Realestate Research Assistant. 
        Your task is to help users by providing accurate and concise information about real estate topics based on their queries.
        
        When responding, ensure that your answers are clear, relevant, and directly address the user's questions.
        Always maintain a professional and helpful tone.
        """
        print("Normal Chat")
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
            contents=conversation_text
        )
        
        ai_message = response.text.strip()
        
        conversation_entry = {
            "user_message": request.message,
            "ai_message": ai_message
        }
        sessions[session_id].append(conversation_entry)
        save_sessions(sessions)
        
    else :
        system_prompt = f"""
        You are a Realestate Assistant. 
        Your task is to help users by providing accurate and concise information about real estate topics based on their queries.
        
        When responding, ensure that your answers are clear, relevant, and directly address the user's questions.
        Always maintain a professional and helpful tone.
        You can use web search results to provide more accurate answers.
        """
        google_search_response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[grounding_tool],
                response_mime_type="application/json",
                response_json_schema=ResponseFormat.model_json_schema()
            ),
            contents=conversation_text
        )
        
        try:
            google_search_response_message = json.loads(
                google_search_response.text.strip()
            )
        except json.JSONDecodeError:
            google_search_response_message = {}

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction="Fetch property data using available tools.",
                tools=[loopnet_tool]
            ),
            contents=conversation_text
        )

        # loopnet_response = gemini_client.models.generate_content(
        #     model=GEMINI_MODEL,
        #     config=types.GenerateContentConfig(
        #         system_instruction="Fetch property data using the provided tool.",
        #         tools=[loopnet_tool]
        #     ),
        #     contents=conversation_text
        # )
        
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                if not part.function_call:
                    continue

                elif part.function_call.name == "loopnet_search_by_city":
                    loopnet__all_data = loopnet_search_by_city(
                        **part.function_call.args
                    )
                    
                    loopnet_properties = loopnet__all_data.get("properties", {}).get("data", [])
                    for listing in loopnet_properties:
                        extracted_listing = {
                            "price": listing.get("price"),
                            "fullPrice": listing.get("fullPrice"),
                            "title": " ".join(listing.get("title", [])) if listing.get("title") else None,
                            "notes": " ".join(listing.get("notes", [])) if listing.get("notes") else None,
                            "photo": listing.get("photo"),
                            "location": listing.get("location", {}).get("name") if listing.get("location") else None,
                            "brokerName": listing.get("brokerName"),
                            "brokerPhoto": listing.get("brokerPhoto")
                        }
                        loopnet_data.append(extracted_listing)

    return {"session_id": session_id, "google_search_response": google_search_response_message, "loopnet_data": loopnet_data, "ai_message": ai_message}