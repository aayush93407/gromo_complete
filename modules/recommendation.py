from flask import request, render_template
import pandas as pd
from catboost import CatBoostClassifier, Pool

# Define categorical columns for the model
cat_features = [
    "Name", "Region", "Language", "Profession",
    "OwnsSmartphone", "MaritalStatus"
]

# Load the CatBoost model once
model = CatBoostClassifier()
model.load_model("modules/catboost_model.cbm")

# Show the recommendation form
def get_upload_form():
    return render_template("recommendation_form.html")

# Handle prediction and show result
def recommend_product(data=None):
    if data is None:
        return "⚠️ No form data submitted."

    try:
        income = float(data.get("Income", "").strip())
        age = int(data.get("Age", "").strip())
        internet_hours = float(data.get("InternetUsageHours", "").strip())
    except (ValueError, TypeError):
        return "⚠️ Invalid numeric input."

    # Prepare input data
    customer_data = pd.DataFrame([{
        "Name": data.get("Name", "").strip(),
        "Region": data.get("Region", "").strip(),
        "Language": data.get("Language", "").strip(),
        "Profession": data.get("Profession", "").strip(),
        "Income": income,
        "Age": age,
        "OwnsSmartphone": data.get("OwnsSmartphone", "").strip(),
        "InternetUsageHours": internet_hours,
        "MaritalStatus": data.get("MaritalStatus", "").strip()
    }])

    input_pool = Pool(customer_data, cat_features=cat_features)
    prediction = model.predict(input_pool)

    # Handle list of list output like [['Savings Account']]
    predicted_label = prediction[0][0] if isinstance(prediction[0], (list, tuple)) else prediction[0]

    return render_template("recommendation_result.html", result=f"🎯 Recommended Product: {predicted_label}")

