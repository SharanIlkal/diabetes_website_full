
Diabetes Prediction — Full working website
=========================================

What's included
- Flask app (app.py) with templates and static assets
- Trained model: diabetes_model_no_preg.pkl (gender-neutral)
- Production-ready files: requirements.txt, Dockerfile, Procfile
- Simple feedback saving (appends to feedback.txt)

Run locally (development)
-------------------------
1. Create virtualenv and install deps:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. Run dev server:
   python app.py
   Open http://127.0.0.1:5000/

Run with Gunicorn (production)
------------------------------
gunicorn -b 0.0.0.0:5000 -w 4 app:app

Docker
------
Build:
  docker build -t diabpredict:latest .
Run:
  docker run -p 8080:8080 diabpredict:latest
Then open http://localhost:8080/

Deploy to Render (example)
--------------------------
- Create a new Web Service on Render using Docker or a Python service.
- If using Docker, push this repo to GitHub and connect Render; it will build the Dockerfile.
- If using Python service, set start command to: gunicorn -b 0.0.0.0:$PORT -w 4 app:app

Notes
-----
This application and model are for demo/educational purposes only and must not be used as a medical diagnosis tool.
