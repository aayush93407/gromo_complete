from flask import Flask, request, jsonify, render_template, render_template_string
from utils.router import route_input
from modules.video import handle_input
from modules.prediction import get_upload_form
import os
import uuid

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
UPLOAD_DIR = "saved"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def ai_agent():
    if request.method == 'POST':
        if 'reference' in request.files or 'video' in request.files:
            return handle_input()

        data = request.get_json(silent=True)
        user_input = data.get("input", "") if data else request.form.get("input", "")
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        output = route_input(user_input)
        if output.strip().startswith("<") and "form" in output:
            return render_template_string(output)
        return jsonify({"result": output})

    return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Agent</title>
  <style>
    body {
      background-color: #e6f0ff; /* light blue */
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      font-family: Arial, sans-serif;
      margin: 0;
    }
    .card {
      background-color: white;
      padding: 30px;
      border-radius: 10px;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
      width: 400px;
      text-align: center;
    }
    input[type="text"] {
      width: 100%;
      padding: 10px;
      margin: 15px 0;
      border: 1px solid #ccc;
      border-radius: 5px;
    }
    input[type="submit"] {
      background-color: #28a745; /* green */
      color: white;
      padding: 10px 20px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      font-size: 16px;
    }
    input[type="submit"]:hover {
      background-color: #218838;
    }
  </style>
</head>
<body>
  <div class="card">
    <h2>AI Agent Test</h2>
    <form method="post" action="/">
      <input type="text" name="input" placeholder="Describe your task" required />
      <input type="submit" value="Submit" />
    </form>
  </div>
</body>
</html>
'''


@app.route('/upload-phone', methods=['GET', 'POST'])
def upload_phone():
    from modules.phone import handle_upload
    return handle_upload()

@app.route("/content-generation", methods=["GET", "POST"])
def content_generation_route():
    from modules import content_generation
    return content_generation.handle_content_generation()

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation_entry():
    from modules import recommendation
    if request.method == 'POST':
        form_data = request.form.to_dict()
        result = recommendation.recommend_product(form_data)
        return render_template('recommendation_result.html', result=result)
    return render_template('recommendation_form.html')

@app.route('/upload-video-face', methods=['GET', 'POST'])
def upload_video_face():
    return handle_input()

# WhatsApp Prediction Upload Page
@app.route('/upload-prediction', methods=['GET'])
def upload_prediction():
    return get_upload_form()

# Handle Prediction Results
from modules.prediction import handle_upload


@app.route("/predict-result", methods=["GET", "POST"])
def predict_result():
    return handle_upload()
    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Render sets PORT env var
    app.run(host='0.0.0.0', port=port, debug=True)


