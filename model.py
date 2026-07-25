from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    result = model.predict([news])[0]

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run()