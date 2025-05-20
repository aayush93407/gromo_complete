import requests
from modules import content_generation, recommendation, prediction, phone, video

MISTRAL_API_KEY = "aKFEMuDwJOvtphHDDOrh2qbfRP7jEA1L"

def get_module_name_from_llm(user_input):
    prompt = f"""
You are an intelligent AI assistant that decides which module to run based on user input.
Choose one of the following module names only:
- content_generation
- recommendation
- prediction
- phone
- video

User input: "{user_input}"

Respond with only the module name.
"""

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip().lower()
    return f"error: {response.text}"

def route_input(user_input):
    module = get_module_name_from_llm(user_input)

    if module == "content_generation":
        return content_generation.handle_content_generation()
    elif module == "recommendation":
        return recommendation.get_upload_form()
    elif module == "prediction":
        return prediction.get_upload_form()
    elif module == "phone":
        return phone.get_upload_form()
    elif module == "video":
        return video.get_upload_form()    
    else:
        return f"❌ Unknown module returned: '{module}'"
