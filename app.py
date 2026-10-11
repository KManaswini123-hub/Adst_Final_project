from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import re
import ast


# =========================================================
# CREATE FLASK APPLICATION
# =========================================================

app = Flask(__name__)

app.secret_key = "medicine_prediction_project_secret_key"


# =========================================================
# LOAD SAVED MACHINE LEARNING FILES
# =========================================================

model = joblib.load("model.pkl")

label_encoder = joblib.load("label_encoder.pkl")

symptom_columns = joblib.load("symptom_columns.pkl")

symptom_mapping = joblib.load("symptom_mapping.pkl")


# =========================================================
# LOAD RECOMMENDATION DATASETS
# =========================================================

medications = pd.read_csv("medications.csv")

description = pd.read_csv("description.csv")

precautions = pd.read_csv("precautions_df.csv")

diets = pd.read_csv("diets.csv")

workout = pd.read_csv("workout_df.csv")


# =========================================================
# LOAD MEDICINE DETAILS DATASET
# =========================================================

medicine_details = pd.read_csv("Medicine_Details.csv")


# =========================================================
# NLP - EXTRACT SYMPTOMS
# =========================================================

def extract_symptoms(user_text):

    text = user_text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    detected_symptoms = []

    for readable_name, original_name in symptom_mapping.items():

        if readable_name in text:

            detected_symptoms.append(original_name)

        elif "fever" in readable_name and "fever" in text:

            detected_symptoms.append(original_name)

    return list(set(detected_symptoms))


# =========================================================
# CREATE MODEL INPUT VECTOR
# =========================================================

def create_symptom_vector(detected_symptoms):

    symptom_vector = [0] * len(symptom_columns)

    for symptom in detected_symptoms:

        if symptom in symptom_columns:

            index = symptom_columns.index(symptom)

            symptom_vector[index] = 1

    return pd.DataFrame(
        [symptom_vector],
        columns=symptom_columns
    )


# =========================================================
# GET DISEASE RECOMMENDATIONS
# =========================================================

def get_recommendations(predicted_disease):

    # -----------------------------------------------------
    # Description
    # -----------------------------------------------------

    description_result = description[
        description["Disease"].astype(str).str.lower()
        == predicted_disease.lower()
    ]

    disease_description = ""

    if not description_result.empty:

        disease_description = description_result[
            "Description"
        ].iloc[0]


    # -----------------------------------------------------
    # Medication
    # -----------------------------------------------------

    medication_result = medications[
        medications["Disease"].astype(str).str.lower()
        == predicted_disease.lower()
    ]

    medicine_info = []

    if not medication_result.empty:

        medicine_value = medication_result[
            "Medication"
        ].iloc[0]

        try:

            medicine_info = ast.literal_eval(
                medicine_value
            )

        except:

            medicine_info = [medicine_value]


    # -----------------------------------------------------
    # Precautions
    # -----------------------------------------------------

    precaution_result = precautions[
        precautions["Disease"].astype(str).str.lower()
        == predicted_disease.lower()
    ]

    precaution_list = []

    if not precaution_result.empty:

        for column in [
            "Precaution_1",
            "Precaution_2",
            "Precaution_3",
            "Precaution_4"
        ]:

            value = precaution_result[
                column
            ].iloc[0]

            if pd.notna(value):

                precaution_list.append(value)


    # -----------------------------------------------------
    # Diet
    # -----------------------------------------------------

    diet_result = diets[
        diets["Disease"].astype(str).str.lower()
        == predicted_disease.lower()
    ]

    diet_info = []

    if not diet_result.empty:

        diet_value = diet_result[
            "Diet"
        ].iloc[0]

        try:

            diet_info = ast.literal_eval(
                diet_value
            )

        except:

            diet_info = [diet_value]


    # -----------------------------------------------------
    # Lifestyle / Workout
    # -----------------------------------------------------

    workout_result = workout[
        workout["disease"].astype(str).str.lower()
        == predicted_disease.lower()
    ]

    workout_list = []

    if not workout_result.empty:

        workout_list = workout_result[
            "workout"
        ].tolist()


    # -----------------------------------------------------
    # Return all information
    # -----------------------------------------------------

    return {

        "description": disease_description,

        "medication": medicine_info,

        "precautions": precaution_list,

        "diet": diet_info,

        "workout": workout_list

    }


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]

        if email and password:

            session["user_email"] = email

            return redirect(
                url_for("dashboard")
            )

    return render_template("login.html")


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]

        email = request.form["email"]

        password = request.form["password"]

        confirm_password = request.form[
            "confirm_password"
        ]

        if password != confirm_password:

            return "Passwords do not match."

        if name and email and password:

            return redirect(
                url_for("login")
            )

    return render_template("register.html")


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.route("/model-info")
def model_info():

    return render_template(
        "model_info.html"
    )


# =========================================================
# MEDICINE INFORMATION
# =========================================================

@app.route("/medicine-info", methods=["GET", "POST"])
def medicine_info():

    medicine = None

    search_query = ""

    if request.method == "POST":

        search_query = request.form.get(
            "medicine",
            ""
        ).strip()

        if search_query:

            # Search medicine name
            result = medicine_details[
                medicine_details[
                    "Medicine Name"
                ].astype(str).str.contains(
                    search_query,
                    case=False,
                    na=False
                )
            ]

            if not result.empty:

                # Display first matching medicine
                medicine = result.iloc[0].to_dict()

    return render_template(
        "medicine_info.html",
        medicine=medicine,
        search_query=search_query
    )


# =========================================================
# RUN PREDICTION FUNCTION
# =========================================================

def run_prediction():

    result = None

    if request.method == "POST":

        user_input = request.form[
            "symptoms"
        ]

        # ---------------------------------------------
        # NLP symptom extraction
        # ---------------------------------------------

        detected_symptoms = extract_symptoms(
            user_input
        )


        # ---------------------------------------------
        # Check whether symptoms were detected
        # ---------------------------------------------

        if len(detected_symptoms) == 0:

            result = {

                "error":
                "No known symptoms were detected. "
                "Please enter symptoms such as fever, "
                "headache, cough, chills, etc."

            }


        else:

            # -----------------------------------------
            # Create ML input
            # -----------------------------------------

            input_vector = create_symptom_vector(
                detected_symptoms
            )


            # -----------------------------------------
            # Predict disease
            # -----------------------------------------

            prediction = model.predict(
                input_vector
            )


            predicted_disease = (
                label_encoder
                .inverse_transform(prediction)[0]
            )


            # -----------------------------------------
            # Get recommendations
            # -----------------------------------------

            recommendations = get_recommendations(
                predicted_disease
            )


            # -----------------------------------------
            # Create final result
            # -----------------------------------------

            result = {

                "input": user_input,

                "symptoms": detected_symptoms,

                "disease": predicted_disease,

                **recommendations

            }

    return result


# =========================================================
# PREDICTION PAGE
# =========================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = run_prediction()

    return render_template(
        "index.html",
        result=result
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = run_prediction()

    return render_template(
        "index.html",
        result=result
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)