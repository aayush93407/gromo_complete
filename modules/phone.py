import os
import uuid
import subprocess
import whisper
import json
import csv
from flask import request, render_template
from mistralai import Mistral

UPLOAD_DIR = "saved"
CSV_FILE = "customer_data.csv"
os.makedirs(UPLOAD_DIR, exist_ok=True)

api_key_mistral = "aKFEMuDwJOvtphHDDOrh2qbfRP7jEA1L"
model_whisper = whisper.load_model("base")

def convert_to_wav(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_audio(audio_path):
    result = model_whisper.transcribe(audio_path)
    return result["text"]

def extract_info_from_text(text):
    prompt = f"""
Extract the following information from the below text:
- Name
- Region
- Language
- Occupation
- Interest

Text:
{text}

Return the result only in this exact JSON format:

{{
  "Name": "...",
  "Region": "...",
  "Language": "...",
  "Occupation": "...",
  "Interest": "..."
}}

All values should be translated into English.
"""
    client = Mistral(api_key=api_key_mistral)
    completion = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    response = completion.choices[0].message.content.strip()

    try:
        json_start = response.index('{')
        json_end = response.rindex('}') + 1
        clean_json = response[json_start:json_end]
        return json.loads(clean_json)
    except ValueError:
        return {}

def get_upload_form():
    return render_template("upload_form.html")

def handle_upload():
    if request.method == "POST":
        file = request.files["audio"]
        ext = os.path.splitext(file.filename)[1].lower()

        filename = f"{uuid.uuid4()}{ext}"
        input_path = os.path.join(UPLOAD_DIR, filename)
        file.save(input_path)

        wav_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
        convert_to_wav(input_path, wav_path)

        transcript = transcribe_audio(wav_path)
        extracted_info = extract_info_from_text(transcript)

        if extracted_info:
            file_exists = os.path.isfile(CSV_FILE)
            with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["Name", "Region", "Language", "Occupation", "Interest"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(extracted_info)

        return render_template("result.html", transcript=transcript, result=extracted_info)
    return get_upload_form()
