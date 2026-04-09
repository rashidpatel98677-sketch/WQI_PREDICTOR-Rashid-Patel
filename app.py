from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load("wqi_regressor.pkl")
scaler = joblib.load("scaler.pkl")

def classify_wqi(score):
    if score >= 95:
        return "EXCELLENT WATER QUALITY (SAFE FOR HUMAN CONSUMPTION)"
    elif score >= 80:
        return "GOOD WATER QUALITY (LIKELY SAFE)"
    elif score >= 60:
        return "MODERATE WATER QUALITY (USE WITH CAUTION)"
    elif score >= 40:
        return "POOR WATER QUALITY (TREATMENT NEEDED)"
    else:
        return "VERY POOR WATER QUALITY (UNSAFE FOR CONSUMPTION)"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_waterQ', methods=['POST'])
def predict_waterQ():
    try:
        # Collect inputs from the user
        inputs = [
            float(request.form["pH"]),
            float(request.form["TDS"]),
            float(request.form["Cl"]),
            float(request.form["SO4"]),
            float(request.form["Na"]),
            float(request.form["K"]),
            float(request.form["Ca"]),
            float(request.form["Mg"]),
            float(request.form["Total_Hardness"])
        ]

        # Standardize the inputs using the loaded scaler
        inputs_scaled = scaler.transform([inputs])

        # Make prediction
        wqi_score = model.predict(inputs_scaled)[0]
        classification = classify_wqi(wqi_score)

        return render_template(
            'result.html',
            wqi_score=round(wqi_score, 2),
            classification=classification
        )
    except Exception as e:
        return f"<h3>Error:</h3><p>{e}</p>"

if __name__ == '__main__':
    app.run(debug=True)
