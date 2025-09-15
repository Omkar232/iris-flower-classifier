FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
# Ensure model is trained when building the image
RUN python train.py
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
