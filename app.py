import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load the models
with open("lung_cancer_model.pkl", "rb") as f:
    lung_model = pickle.load(f)

with open("drug_response_model.pkl", "rb") as f:
    drug_model = pickle.load(f)

# Lung Cancer Input & Prediction
def predict_lung_cancer(GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
                        CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING,
                        COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN):
    
    gender = 1 if GENDER == "M" else 0

    features = [[
        gender, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
        CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING,
        COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN
    ]]

    prediction = lung_model.predict(features)[0]
    
    if prediction == "YES":
        return "⚠️ Lung Cancer Risk Detected. Please continue to drug response check.", gr.update(visible=True)
    else:
        return "✅ Low Risk of Lung Cancer.", gr.update(visible=False)

# Drug Response Prediction
def predict_drug_response(age, sex, weight, blood_pressure, cholesterol, glucose,
                          genetic_marker_1, genetic_marker_2, drug_dosage, drug_duration,
                          previous_conditions, liver_function_score):
    
    features = [[
        age, sex, weight, blood_pressure, cholesterol, glucose,
        genetic_marker_1, genetic_marker_2, drug_dosage, drug_duration,
        previous_conditions, liver_function_score
    ]]
    
    prediction = drug_model.predict(features)[0]
    
    return "✅ Likely to Respond to Treatment" if prediction == 1 else "❌ Unlikely to Respond"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Lung Cancer & Drug Response Prediction")
    gr.Markdown("### Step 1: Lung Cancer Risk Check")

    with gr.Row():
        GENDER = gr.Radio(["M", "F"], label="Gender")
        AGE = gr.Number(label="Age")
        SMOKING = gr.Slider(0, 2, label="Smoking (0=No, 2=High)")
        YELLOW_FINGERS = gr.Slider(0, 2, label="Yellow Fingers")
        ANXIETY = gr.Slider(0, 2, label="Anxiety")
        PEER_PRESSURE = gr.Slider(0, 2, label="Peer Pressure")
        CHRONIC_DISEASE = gr.Slider(0, 2, label="Chronic Disease")
        FATIGUE = gr.Slider(0, 2, label="Fatigue")
        ALLERGY = gr.Slider(0, 2, label="Allergy")
        WHEEZING = gr.Slider(0, 2, label="Wheezing")
        ALCOHOL_CONSUMING = gr.Slider(0, 2, label="Alcohol Consuming")
        COUGHING = gr.Slider(0, 2, label="Coughing")
        SHORTNESS_OF_BREATH = gr.Slider(0, 2, label="Shortness of Breath")
        SWALLOWING_DIFFICULTY = gr.Slider(0, 2, label="Swallowing Difficulty")
        CHEST_PAIN = gr.Slider(0, 2, label="Chest Pain")
    
    lung_output = gr.Textbox(label="Lung Cancer Result")
    next_section = gr.Column(visible=False)

    lung_button = gr.Button("🔍 Predict Lung Cancer")
    lung_button.click(
        fn=predict_lung_cancer,
        inputs=[
            GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
            CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING,
            COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN
        ],
        outputs=[lung_output, next_section]
    )

    with next_section:
        gr.Markdown("### Step 2: Drug Response Check")

        with gr.Row():
            age = gr.Number(label="Age")
            sex = gr.Number(label="Sex (1=Male, 0=Female)")
            weight = gr.Number(label="Weight")
            blood_pressure = gr.Number(label="Blood Pressure")
            cholesterol = gr.Number(label="Cholesterol")
            glucose = gr.Number(label="Glucose")
            genetic_marker_1 = gr.Number(label="Genetic Marker 1")
            genetic_marker_2 = gr.Number(label="Genetic Marker 2")
            drug_dosage = gr.Number(label="Drug Dosage")
            drug_duration = gr.Number(label="Drug Duration")
            previous_conditions = gr.Number(label="Previous Conditions Score")
            liver_function_score = gr.Number(label="Liver Function Score")

        drug_output = gr.Textbox(label="Drug Response Result")
        drug_button = gr.Button("💊 Predict Drug Response")
        drug_button.click(
            fn=predict_drug_response,
            inputs=[
                age, sex, weight, blood_pressure, cholesterol, glucose,
                genetic_marker_1, genetic_marker_2, drug_dosage, drug_duration,
                previous_conditions, liver_function_score
            ],
            outputs=drug_output
        )

demo.launch()
