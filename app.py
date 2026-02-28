from flask import Flask, render_template, request
from src.inference import predict
import pandas as pd
from src.evaluate_models import evaluate_models

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        result = predict(text)
    return render_template("index.html", result=result)

@app.route("/admin")
def admin():
    complaints = pd.read_csv("data/raw/complaints.csv").to_dict(orient="records")
    metrics = evaluate_models()

    return render_template(
        "admin.html",
        complaints=complaints,
        metrics=metrics
    )

if __name__ == "__main__":
    app.run(debug=True)