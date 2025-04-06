# 🧠 AI-Powered Health Guardian – Training Framework (Work-in-Progress)

## 📌 Description

This project addresses the need for secure and user-focused solutions that transform wearable data into personalised, actionable health insights. By developing an agent driven by a Large Language Model (LLM), this project aims to provide personalised health analysis while prioritising privacy. The goal is to combine techniques like Federated Learning and Differential Privacy into the agent to ensure users’ data is analysed securely and their privacy is protected.  
The objectives of this project include evaluating existing privacy-preserving methods detailed above (FL and DP), and assessing their applicability to healthcare data from wearables and the integration of these into LLMs. The project aims to discover creative ways to provide personalised insights while maintaining privacy.  
Completion will be measured by developing a model or prototype and assessing its accuracy and privacy capabilities compared to benchmarks or baselines.  

## 🛠️ Contents

> 🔧 *Training-focused phase – hardware integration and deployment components will be added in future updates.*  
├── data_preprocessing/ # Scripts for cleaning and formatting health data  
> ├── models/ # Model architectures (e.g., LSTM, CNN, Transformer)  
> ├── trainers/ # Training and evaluation scripts  
> ├── experiments/ # Experiment configs and results  
> ├── utils/ # Helper functions (metrics, logging, etc.)  
> ├── datasets/ #The dataset used for training after cleaning  
> └── README.md # Project overview  

## 🧠 Training Methods

Implemented training strategies include:  
- Supervised learning on time-series physiological data  
- Sequence modeling using sliding windows  
- Evaluation metrics: accuracy, F1-score, precision, recall  
- Support for differential privacy & federated learning & Secure Aggregation  

## 🧰 Requirements

- Python 3.9+  
- PyTorch or TensorFlow  
- NumPy, Pandas, Scikit-learn  
- (Optional) Weights & Biases for experiment tracking  


