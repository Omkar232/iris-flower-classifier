from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

MODEL_PATH = "model.pkl"

app = Flask(__name__)

# Minimal HTML template
HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Iris Classifier</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; }
      input { width: 100%; padding: 8px; margin: 8px 0; }
      button { padding: 10px 16px; }
      .row { display:flex; gap:10px; }
      .col { flex:1 }
    </style>
  </head>
  <body>
    <h1>Iris Flower Classifier</h1>
    <p>Enter four features (sepal length, sepal width, petal length, petal width)</p>
    <form id="predict-form">
      <div class="row">
        <input name="f0" placeholder="sepal length (e.g. 5.1)" />
        <input name="f1" placeholder="sepal width (e.g. 3.5)" />
      </div>
      <div class="row">
        <input name="f2" placeholder="petal length (e.g. 1.4)" />
        <input name="f3" placeholder="petal width (e.g. 0.2)" />
      </div>
      <button type="button" onclick="send()">Predict</button>
    </form>
    <h2 id="result"></h2>

    <script>
      async function send(){
        const form = document.getElementById('predict-form');
        const data = [
          parseFloat(form.f0.value),
          parseFloat(form.f1.value),
          parseFloat(form.f2.value),
          parseFloat(form.f3.value)
        ];
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features: data })
        });
        const j = await resp.json();
        if (j.error) {
          document.getElementById('result').innerText = 'Error: ' + j.error;
        } else {
          document.getElementById('result').innerText = j.class_name + ' (class ' + j.class_id + ')';
        }
      }
    </script>
  </body>
</html>
"""

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found. Run `python train.py` to create {MODEL_PATH}")
    data = joblib.load(MODEL_PATH)
    return data['model'], data['target_names']

model, target_names = None, None

@app.before_first_request
def startup():
    global model, target_names
    model, target_names = load_model()

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    global model, target_names
    payload = request.get_json(force=True)
    if not payload or 'features' not in payload:
        return jsonify({'error': 'Payload must be JSON with key \"features\" (list of 4 floats)'}), 400

    features = payload['features']
    try:
        arr = np.array(features, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({'error': 'Features must be a list of 4 numbers.'}), 400

    pred = model.predict(arr)[0]
    class_name = target_names[int(pred)]
    return jsonify({'class_id': int(pred), 'class_name': class_name})

if __name__ == '__main__':
    # For production, use gunicorn / a proper WSGI server
    app.run(host='0.0.0.0', port=5000, debug=True)
