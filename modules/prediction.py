import os
import uuid
import traceback
import pandas as pd
import spacy
from flask import request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
from whatstk import df_from_whatsapp
import openai

# Directory to save uploaded files
UPLOAD_DIR = "saved"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Use environment variable for API key (fallback to default only in dev)
api_key_mistral = os.getenv("MISTRAL_API_KEY", "aKFEMuDwJOvtphHDDOrh2qbfRP7jEA1L")

# Load NLP models once
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Keywords for interest classification
interest_keywords = ["interest", "want", "apply", "yes", "sure", "how", "detail", "send", "okay", "please", "share", "process", "offer"]
disinterest_keywords = ["not interest", "no", "don't want", "not now", "maybe later", "stop", "never", "don't call", "not require", "not need"]

# Detect interest using simple NLP keyword/lemma matching
def detect_interest_nlp(text):
    doc = nlp(str(text).lower())
    lemmas = " ".join([token.lemma_ for token in doc])
    for word in interest_keywords:
        if word in lemmas:
            return "Interested"
    for word in disinterest_keywords:
        if word in lemmas:
            return "Not Interested"
    return "Neutral"

# Analyze WhatsApp chat
def analyze_whatsapp_chat(input_path):
    df = df_from_whatsapp(input_path)

    if df.empty or 'message' not in df.columns or 'username' not in df.columns:
        raise ValueError("Invalid or empty WhatsApp chat file.")

    # Sentiment analysis
    def get_sentiment(text):
        result = sentiment_analyzer(str(text))[0]
        return result['label'], result['score']

    df[['sentiment', 'sentiment_score']] = df['message'].apply(lambda x: pd.Series(get_sentiment(x)))
    df['interest_level_nlp'] = df['message'].apply(detect_interest_nlp)

    # Train interest classifier
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['interest_level_nlp']
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    df['predicted_interest'] = clf.predict(vectorizer.transform(df['message']))

    return df

# Summary based on NLP classification
def generate_summary(df):
    interest_counts = df['interest_level_nlp'].value_counts().to_dict()
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    interested = interest_counts.get('Interested', 0)
    not_interested = interest_counts.get('Not Interested', 0)
    neutral = interest_counts.get('Neutral', 0)

    if interested > max(not_interested, neutral):
        interest_summary = "The customer is interested to buy."
        interest_score = 2
    elif not_interested > max(interested, neutral):
        interest_summary = "The customer is not interested to buy."
        interest_score = 0
    else:
        interest_summary = "The customer's interest is neutral or unclear."
        interest_score = 1

    pos = sentiment_counts.get('Positive', 0)
    neg = sentiment_counts.get('Negative', 0)
    neu = sentiment_counts.get('Neutral', 0)

    if pos > max(neg, neu):
        sentiment_summary = "Overall sentiment is positive."
        sentiment_score = 2
    elif neg > max(pos, neu):
        sentiment_summary = "Overall sentiment is negative."
        sentiment_score = 0
    else:
        sentiment_summary = "Overall sentiment is neutral."
        sentiment_score = 1

    # Calculate overall score
    total_score = interest_score + sentiment_score
    if total_score >= 3:
        overall_result = "Satisfied"
    else:
        overall_result = "Normal"

    final_summary = f"{interest_summary} {sentiment_summary}"

    return final_summary, overall_result
# Generate feedback for seller using Mistral
def get_seller_feedback(chat_df, seller_name, api_key_mistral):
    seller_msgs = chat_df[chat_df['username'] == seller_name]['message'].tolist()
    if not seller_msgs:
        return "No seller messages found in the chat."

    chat_text = "\n".join(seller_msgs[-15:])  # Last 15 messages for context

    prompt = (
        "You are a sales coach. Analyze the following WhatsApp sales messages sent by a seller to a customer. "
        "Give actionable, constructive feedback on how the seller can improve their sales pitch.\n\n"
        f"Seller's messages:\n{chat_text}\n\n"
        "Suggestions:"
    )

    client = openai.OpenAI(
        api_key=api_key_mistral,
        base_url="https://api.mistral.ai/v1"
    )

    response = client.chat.completions.create(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# Serve upload form
def get_upload_form():
    return render_template("upload_whatsapp.html")

# Handle form POST and process chat
def handle_upload():
    if request.method == "POST":
        try:
            seller_name = request.form.get("seller_name", "").strip()
            api_key = request.form.get("api_key_mistral", "").strip()
            file = request.files.get("chat")

            if not file or not api_key:
                return render_template("error.html", message="Missing chat file or API key.")

            # Save uploaded file
            filename = f"{uuid.uuid4()}.txt"
            file_path = os.path.join(UPLOAD_DIR, filename)
            file.save(file_path)

            # Analyze and summarize
            chat_df = analyze_whatsapp_chat(file_path)
            summary, overall_result = generate_summary(chat_df)

            feedback = get_seller_feedback(chat_df, seller_name, api_key)

            return render_template(
                "whatsapp_result.html",
                summary=summary,
                feedback=feedback,
                overall_result=overall_result,
                chat_html=chat_df.to_html(classes="table table-bordered", index=False)
            )

        except Exception as e:
            traceback.print_exc()
            return render_template("error.html", message=f"An error occurred: {str(e)}")

    return get_upload_form()
