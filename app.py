from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("models/model.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        Type = float(request.form.get("Type"))
        Amount = float(request.form.get("Amount"))
        Old_Balance = float(request.form.get("Old Balance"))
        New_Balance = float(request.form.get("New Balance"))

        new_data_scaled = standard_scaler.transform(
            [[Type, Amount, Old_Balance, New_Balance]]
        )
        predict = model.predict(new_data_scaled)[0]
        if predict == 1:
            result = "Fraud"
        else:
            result = "Not Fraud"

        return render_template("index.html", result=predict)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug="True")
