from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

# Create Flask app
app = Flask(__name__, template_folder="../templates")

# Get current directory (app folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Load Model
# ----------------------------
model_path = os.path.join(BASE_DIR, "..", "models", "random_forest.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Load Training Columns
# ----------------------------
xtrain_path = os.path.join(BASE_DIR, "..", "notebook", "X_train.csv")
X_train = pd.read_csv(xtrain_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ----------------------------
        # Get User Inputs
        # ----------------------------
        total_exp = float(request.form["total_exp"])
        field_exp = float(request.form["field_exp"])
        degree = request.form["degree"]
        skills = request.form.getlist("skills")
        job_role = request.form["job_role"]
        industry = request.form["industry"]

        # ----------------------------
        # Create Empty Input DataFrame
        # ----------------------------
        df = pd.DataFrame(columns=X_train.columns)
        df.loc[0] = 0

        # ----------------------------
        # Numeric Features
        # ----------------------------
        if "Total_Experience" in df.columns:
            df["Total_Experience"] = total_exp

        if "Total_Experience_in_field_applied" in df.columns:
            df["Total_Experience_in_field_applied"] = field_exp

        if "exp_ratio" in df.columns:
            df["exp_ratio"] = field_exp / (total_exp + 1)

        # ----------------------------
        # Degree Encoding
        # ----------------------------
        degree_col = f"Degree_{degree}"
        if degree_col in df.columns:
            df[degree_col] = 1

        # ----------------------------
        # Skills Encoding
        # ----------------------------
        for skill in skills:
            skill_col = f"Skills_{skill}"
            if skill_col in df.columns:
                df[skill_col] = 1

        # ----------------------------
        # Job Role Encoding
        # ----------------------------
        job_col = f"Job_Role_{job_role}"
        if job_col in df.columns:
            df[job_col] = 1

        # ----------------------------
        # Industry Encoding
        # ----------------------------
        industry_col = f"Industry_{industry}"
        if industry_col in df.columns:
            df[industry_col] = 1

        # ----------------------------
        # Prediction
        # ----------------------------
        prediction = model.predict(df)[0]
        ctc = np.expm1(prediction)

        # Experience adjustments
        ctc *= (1 + 0.10 * total_exp)
        ctc *= (1 + 0.05 * field_exp)
        ctc *= (1 + 0.02 * len(skills))

        # Job Role Bonus
        if job_role == "Data Scientist":
            ctc *= 1.20
        elif job_role == "Project Manager":
            ctc *= 1.15

        # Industry Bonus
        if industry == "Finance":
            ctc *= 1.10
        elif industry == "Healthcare":
            ctc *= 1.05

        ctc_lpa = round(ctc / 100000, 2)

        return render_template(
            "index.html",
            prediction_text=f"Expected CTC: ₹{ctc_lpa} LPA"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"⚠️ Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)