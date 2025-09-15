# Iris Flower Classifier — Web App

A simple AI/ML project that trains a scikit-learn model on the Iris dataset and serves predictions via a lightweight Flask web app.

## Features
- `train.py` trains a RandomForest model and saves `model.pkl`.
- `app.py` runs a Flask app with:
  - Web UI for manual input
  - JSON API endpoint `POST /predict` for programmatic access
- `requirements.txt` lists dependencies
- `Dockerfile` to containerize the app

## Quick start (local)
1. Create virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
venv\\Scripts\\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train.py
```

This will create `model.pkl` in the project folder.

4. Run the web app:

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

## Quick test with curl

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Docker

Build and run with Docker:

```bash
docker build -t iris-classifier:latest .
docker run -p 5000:5000 iris-classifier:latest
```

## Files in this repo
- `train.py` — training script
- `app.py` — Flask web app
- `requirements.txt`
- `Dockerfile`
- `.gitignore`
- `README.md`

## License
MIT
