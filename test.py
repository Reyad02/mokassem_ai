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

GEMINI_MODEL = "gemini-3-flash-preview"

def extract_property_data(property_data: dict) -> dict:
    return {
        "price": property_data.get("price"),

        "name": property_data.get("title"),

        "description": (
            f"Luxury residential apartment located in "
            f"{property_data['location']['community']['name']}, "
            f"{property_data['location']['city']['name']} with modern amenities "
            f"and excellent connectivity."
        ),

        "amenities": property_data.get("amenities", []),

        "broker": {
            "name": property_data.get("agent", {}).get("name"),
            "agency": property_data.get("agency", {}).get("name")
        },

        "contact": {
            "mobile": property_data.get("agent", {}).get("contact", {}).get("mobile"),
            "phone": property_data.get("agent", {}).get("contact", {}).get("phone"),
            "whatsapp": property_data.get("agent", {}).get("contact", {}).get("whatsapp")
        },

        "property_details": {
            "bedrooms": property_data.get("details", {}).get("bedrooms"),
            "bathrooms": property_data.get("details", {}).get("bathrooms"),
            "sqft": property_data.get("area", {}).get("built_up"),
            "furnished": property_data.get("details", {}).get("is_furnished"),
            "completion_status": property_data.get("details", {}).get("completion_status")
        },

        "images": property_data.get("media", {}).get("photos", [])[:5]
    }

def fetch_uae_properties(payload: dict):
    # ✅ Default purpose
    payload.setdefault("purpose", "for-sale")

    url = "https://uae-real-estate2.p.rapidapi.com/properties_search"

    querystring = {
        "page": "0",
        "langs": "en"
    }

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "uae-real-estate2.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        params=querystring,
        timeout=30
    )

    response.raise_for_status()
    return response.json()

class PropertyDetails(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft: int

class BrokerInfo(BaseModel):
    name: str
    agency: str
    
class ContactInfo(BaseModel):
    mobile: str | None = None
    phone: str | None = None
    whatsapp: str | None = None
    
class ResponseFormat(BaseModel):
    price: str
    name: str
    property_details: List[PropertyDetails]
    amenities: List[str]
    description: str
    # about: str
    # image_urls: List[str]
    broker: BrokerInfo
    contact: ContactInfo

class ResearchChat(BaseModel):
    session_id: str | None = None
    message: str
    type: str 

uae_property_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="fetch_uae_properties",
            description="Search UAE real estate properties. If purpose is not provided, assume 'for-sale'.",
            parameters={
                "type": "object",
                "properties": {
                    "purpose": {
                        "type": "string",
                        "default": "for-sale"
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["apartments", "villas"]},
                    },
                    "price_min": {"type": "number"},
                    "price_max": {"type": "number"},
                    "rooms": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "baths": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "area_min": {"type": "number"},
                    "area_max": {"type": "number"},
                    "amenities": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        )
    ]
)


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
    uae_properties = []
   
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
            
        uae_properties_response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction="Fetch property data using available tools.",
                tools=[uae_property_tool],
                response_mime_type="application/json",
                response_json_schema=ResponseFormat.model_json_schema()
            ),
            contents=conversation_text
        )
        
        if uae_properties_response.candidates:
            for part in uae_properties_response.candidates[0].content.parts:
                if part.function_call:
                    fn_name = part.function_call.name
                    fn_args = part.function_call.args

                    if fn_name == "fetch_uae_properties":
                        uae_properties_data = fetch_uae_properties(fn_args)
                        for i, prop in enumerate(uae_properties_data.get("results", [])):
                            # uae_properties["results"][i] = extract_property_data(prop)
                            uae_properties.append(extract_property_data(prop))

    return {"session_id": session_id, "google_search_response": google_search_response_message, "uae_properties": uae_properties, "ai_message": ai_message}