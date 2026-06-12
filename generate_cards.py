import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load your secure keys from the .env file
load_dotenv()

# 2. Initialize the client to point at OpenRouter instead of OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def generate_learning_chain(topic: str):
    print(f"Generating learning chain for: '{topic}'...\n")
    
    # 3. The System Prompt (This is the secret sauce for NeuronDeck)
    # We heavily restrict the AI to act ONLY as a JSON data compiler.
    system_prompt = """
    You are the core intelligence of NeuronDeck, a structured educational engine.
    Your ONLY job is to take a user's topic and break it down into a logical, sequential chain of learning cards.
    
    RULES:
    1. NEVER include greetings, pleasantries, or introductory text (e.g., "Here are your cards", "Sure!").
    2. NEVER output markdown wrapping the JSON (no ```json).
    3. Output ONLY a valid JSON array of objects.
    4. Generate exactly 5 to 8 cards in strict educational progression order.
    
    SCHEMA FOR EACH CARD (JSON):
    {
        "title": "Short title (e.g. What is X?)",
        "shortText": "A 5-8 word preview of the concept.",
        "fullText": "The complete, concise explanation. Use bullet points or formulas if necessary.",
        "sequence": 1 // (Must increment by 1 for each card)
    }
    """

    try:
        # 4. Call the Free OpenRouter API
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Teach me about: {topic}"}
            ],
            # OpenRouter specific headers recommended for tracking
            extra_headers={
                "HTTP-Referer": "http://localhost:3000", 
                "X-Title": "NeuronDeck Local Testing",
            }
        )

        # 5. Extract and format the output
        raw_output = response.choices[0].message.content
        
        # Try to parse the text into an actual Python dictionary to verify it's valid JSON
        cards_data = json.loads(raw_output)
        
        # Print it beautifully to the terminal
        print("Success! Here is the JSON output ready for your Vanilla JS frontend:\n")
        print(json.dumps(cards_data, indent=2))
        
        return cards_data

    except json.JSONDecodeError:
        print("ERROR: The AI did not return valid JSON. It probably included conversational filler.")
        print("Raw output:")
        print(raw_output)
    except Exception as e:
        print(f"An API error occurred: {e}")

# 6. Run the test!
if __name__ == "__main__":
    test_topic = "Probability Basics"
    generate_learning_chain(test_topic)