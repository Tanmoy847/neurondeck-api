import os
import json
import re
import urllib.parse
import requests
import chromadb
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load local .env file if running locally
load_dotenv()

# ==========================================
# CONFIGURATION & SECRETS
# ==========================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

chroma_client = chromadb.PersistentClient(path="./neuron_memory")

app = FastAPI()

# ---------------------------------------------------------
# ENDPOINT 0: DOUBLE HEARTBEAT
# ---------------------------------------------------------
@app.get("/")
def keep_alive():
    try:
        requests.get(
            f"{SUPABASE_URL}/rest/v1/private_card_versions?limit=1",
            headers={"apikey": SUPABASE_ANON_KEY}
        )
    except Exception as e:
        print(f"Heartbeat DB Ping failed: {e}")
        
    return {"status": "NeuronDeck Engine & Database are Awake"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"], 
)

class TopicRequest(BaseModel):
    topic: str
    start_sequence: int = 1

class ClarifyRequest(BaseModel):
    topic: str
    card_title: str
    current_text: str
    sequence: int

def verify_user(authorization: str):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Token")
    token = authorization.split(" ")[1]
    auth_response = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY}
    )
    if auth_response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Security Token")
    return auth_response.json().get("id")

# ---------------------------------------------------------
# ENDPOINT 1: CORE ENGINE & PAGINATION
# ---------------------------------------------------------
@app.post("/api/learn")
async def generate_learning_chain(request: TopicRequest, authorization: str = Header(None)):
    user_id = verify_user(authorization)
    
    global_memory = chroma_client.get_or_create_collection(name="global_knowledge_base")
    clean_topic = request.topic.lower().strip()
    storage_id = f"{clean_topic}_batch_{request.start_sequence}"

    existing_results = global_memory.get(ids=[storage_id])
    
    if existing_results and existing_results.get('documents') and len(existing_results['documents']) > 0:
        print(f"Serving chunk from global database index: {storage_id}")
        base_cards = json.loads(existing_results['documents'][0])
    else:
        print(f"Generating fresh cards from index {request.start_sequence} for: {clean_topic}...")
        
        prompt = f"""
        You are an expert university professor and strict academic tutor. 
        Create a highly accurate, strictly sequenced learning chain about: "{request.topic}".
        
        CRITICAL RULES:
        - Start numbering card sequences from number: {request.start_sequence}.
        - Generate exactly 4 progressive cards moving forward logically.
        - NEVER use unescaped double quotes inside your text values. Use single quotes for names/terms (e.g. 'AND' gate, 'NOT' gate).
        - Respond ONLY with a valid JSON object matching this schema:
        {{
          "cards": [
            {{
              "sequence": {request.start_sequence},
              "title": "Card Title",
              "shortText": "A one-sentence summary.",
              "fullText": "A detailed 2-3 paragraph explanation. Use \\n for line breaks."
            }}
          ]
        }}
        """

        try:
            openrouter_response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://neurondeck.vercel.app",
                    "X-Title": "NeuronDeck"
                },
                json={
                    "model": "google/gemma-4-26b-a4b-it:free",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }
            )
            
            response_json = openrouter_response.json()
            
            if "error" in response_json:
                error_msg = response_json['error'].get('message', str(response_json['error']))
                print(f"OpenRouter API Blocked Request: {error_msg}")
                raise Exception(f"AI Error: {error_msg}")

            ai_output = response_json['choices'][0]['message']['content']
            
            # Clean potential markdown code blocks
            clean_output = re.sub(r'^```json\s*|^```\s*|```$', '', ai_output.strip(), flags=re.MULTILINE)
            match = re.search(r'\{.*\}', clean_output, re.DOTALL)
            if not match:
                raise Exception("No valid JSON object found in response.")
            
            clean_json_string = match.group(0)
            new_chain_data = json.loads(clean_json_string, strict=False)
            
            base_cards = new_chain_data['cards']
            
            global_memory.add(
                documents=[json.dumps(base_cards)], 
                metadatas=[{"topic": clean_topic, "start": request.start_sequence}], 
                ids=[storage_id]
            )
            
        except Exception as e:
            print(f"Engine Error: {str(e)}")
            if 'ai_output' in locals():
                print(f"Raw Output Received:\n{ai_output}")
            raise HTTPException(status_code=500, detail="Failed to process knowledge chain cycle.")

    # Inject Private User Versions
    token = authorization.split(" ")[1]
    get_url = f"{SUPABASE_URL}/rest/v1/private_card_versions?user_id=eq.{user_id}"
    db_response = requests.get(get_url, headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"})
    
    if db_response.status_code == 200:
        all_user_records = db_response.json()
        private_overrides = [r for r in all_user_records if r['topic'] == clean_topic]
        
        for card in base_cards:
            card['versions'] = [{
                "title": card['title'], 
                "shortText": card['shortText'], 
                "fullText": card['fullText']
            }]
            
            for override in private_overrides:
                if override['sequence_id'] == card['sequence']:
                    card['versions'] = override['versions']
                    
            card['current_version_index'] = len(card['versions']) - 1

    return {"cards": base_cards}

# ---------------------------------------------------------
# ENDPOINT 2: CONFUSION ENGINE
# ---------------------------------------------------------
@app.post("/api/clarify")
async def clarify_card(request: ClarifyRequest, authorization: str = Header(None)):
    user_id = verify_user(authorization)
    token = authorization.split(" ")[1]
    clean_topic = request.topic.lower().strip()
    clean_title = request.card_title.replace(" (Simplified)", "")
    
    print(f"Confusion Mode Triggered for Card {request.sequence}: {clean_title}")

    prompt = f"""
    A user is confused by a flashcard about the topic: "{request.topic}".
    The card title is: "{clean_title}".
    The current explanation: "{request.current_text}".
    
    TASK:
    Regenerate content for THIS SPECIFIC CARD ONLY with intuitive real-world analogies.
    
    CRITICAL RULES:
    - NEVER use unescaped double quotes inside your text values. Use single quotes instead.
    - Respond ONLY with a valid JSON object matching this schema:
    {{
      "sequence": {request.sequence},
      "title": "{clean_title} (Simplified)",
      "shortText": "An intuitive one-sentence summary.",
      "fullText": "New explanation using simple analogies. Use \\n for line breaks."
    }}
    """

    try:
        openrouter_response = requests.post(
            url="[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "google/gemma-4-26b-a4b-it:free",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
        )
        
        response_json = openrouter_response.json()
        if "error" in response_json:
            raise Exception(f"AI Provider Error: {response_json['error'].get('message', str(response_json['error']))}")
            
        ai_output = response_json['choices'][0]['message']['content']
        clean_output = re.sub(r'^```json\s*|^```\s*|```$', '', ai_output.strip(), flags=re.MULTILINE)
        match = re.search(r'\{.*\}', clean_output, re.DOTALL)
        refinedCard = json.loads(match.group(0), strict=False) 
        
    except Exception as e:
        print(f"Confusion Engine Error: {str(e)}")
        if 'ai_output' in locals():
            print(f"Raw Output Received:\n{ai_output}")
        raise HTTPException(status_code=500, detail="Failed to clarify content.")

    # Supabase Version Persistence
    get_url = f"{SUPABASE_URL}/rest/v1/private_card_versions?user_id=eq.{user_id}&sequence_id=eq.{request.sequence}"
    existing_versions_req = requests.get(get_url, headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"})
    
    version_history = []
    row_id = None
    
    if existing_versions_req.status_code == 200:
        records = existing_versions_req.json()
        existing_row = next((r for r in records if r['topic'] == clean_topic), None)
        if existing_row:
            version_history = existing_row['versions']
            row_id = existing_row['id']

    if not version_history:
        version_history.append({
            "title": request.card_title,
            "shortText": "Original Explanation",
            "fullText": request.current_text
        })
        
    version_history.append({
        "title": refinedCard["title"],
        "shortText": refinedCard["shortText"],
        "fullText": refinedCard["fullText"]
    })
    
    if row_id:
        db_res = requests.patch(
            f"{SUPABASE_URL}/rest/v1/private_card_versions?id=eq.{row_id}",
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"versions": version_history}
        )
    else:
        db_res = requests.post(
            f"{SUPABASE_URL}/rest/v1/private_card_versions",
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "user_id": user_id,
                "topic": clean_topic,
                "sequence_id": request.sequence,
                "versions": version_history
            }
        )

    if db_res.status_code not in (200, 201, 204):
        error_msg = db_res.text
        print(f"\nCRITICAL DATABASE BLOCK: {error_msg}\n")
        raise HTTPException(status_code=500, detail=f"Database rejected save: {error_msg}")

    return {"versions": version_history, "current_version_index": len(version_history) - 1}
