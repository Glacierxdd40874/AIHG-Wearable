import gradio as gr
import requests
import re
import numpy as np
from keras.models import load_model

# Paths to model and normalization files
MODEL_PATH = "C:/Users/Xavierxdd/Desktop/final_model.h5"
MEAN_PATH = "C:/Users/Xavierxdd/Desktop/mean.npy"
STD_PATH = "C:/Users/Xavierxdd/Desktop/std.npy"

# Load Keras model and normalization parameters
model = load_model(MODEL_PATH)
mean = np.load(MEAN_PATH)
std = np.load(STD_PATH)

# Ollama local API setup
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"  # Change to your actual model name if needed

def query_llm(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]

def parse_user_input(text: str):
    prompt = f"""You are a wellness assistant. Convert the following description into 5 health scores with the following ranges:
- fatigue: 0–5
- mood: 0–5
- readiness: 0–10
- soreness: 0–5
- stress: 0–5

Respond strictly in this format:
fatigue: x  
mood: x  
readiness: x  
soreness: x  
stress: x

Text:
{text}
"""
    result = query_llm(prompt)
    numbers = list(map(float, re.findall(r"\d+(?:\.\d+)?", result)))
    if len(numbers) != 5:
        raise ValueError("Failed to extract 5 health scores from the LLM response.")
    return dict(zip(["fatigue", "mood", "readiness", "soreness", "stress"], numbers))

def predict_sleep(features_dict):
    input_data = np.array([[features_dict[key] for key in ["fatigue", "mood", "readiness", "soreness", "stress"]]])
    normalized = (input_data - mean) / std
    prediction = model.predict(normalized)[0]

    sleep_duration = float(np.clip(prediction[0], 0, 12))
    sleep_quality = float(np.clip(prediction[1], 0, 5))
    return round(sleep_duration, 2), round(sleep_quality, 2)

def generate_advice(sleep_duration, sleep_quality):
    prompt = f"""You are a health advisor. Based on the user's predicted sleep indicators:
- Sleep Duration: {sleep_duration} hours
- Sleep Quality: {sleep_quality} / 5

Please provide 2–3 personalized recommendations to improve the user's sleep and recovery.
"""
    return query_llm(prompt)

def analyze_health(user_input):
    try:
        scores = parse_user_input(user_input)
        sleep_duration, sleep_quality = predict_sleep(scores)
        advice = generate_advice(sleep_duration, sleep_quality)

        result_summary = (
            f"###  LLM Interpretation of Input:\n"
            f"- Fatigue: {scores['fatigue']}\n"
            f"- Mood: {scores['mood']}\n"
            f"- Readiness: {scores['readiness']}\n"
            f"- Soreness: {scores['soreness']}\n"
            f"- Stress: {scores['stress']}\n\n"
            f"###  Model Predictions:\n"
            f"- Sleep Duration: {sleep_duration} hours\n"
            f"- Sleep Quality: {sleep_quality} / 5\n\n"
            f"###  Recommendations:\n{advice.strip()}"
        )

        return result_summary
    except Exception as e:
        return f" Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=analyze_health,
    inputs=gr.Textbox(lines=4, label="Describe how you're feeling (stress, fatigue, etc.)"),
    outputs=gr.Markdown(label="Results and Recommendations"),
    title=" Sleep & Wellness Advisor",
    description="Enter a brief description of your recent well-being. This app will analyze your input, predict your sleep, and provide tips.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
