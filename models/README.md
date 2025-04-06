# Models

This directory contains the implementations of various machine learning models used for the **AI-Powered Health Guardian** project. These models are designed to analyze time-series data (such as physiological data from wearable IoT devices) to predict health conditions, detect anomalies, and generate personalized health insights.  

## Models

### 1. **Federated Learning Wrappers (`Flower/`)**
This module includes the necessary wrappers and adapters to integrate the models with federated learning frameworks **Flower** .  
- **Input**: Local datasets on different devices
- **Output**:
  - Federated model updates aggregated across devices
  - Accuracy, Training time, Loss and Memory Usage of the framework
- **Features**:
  - Ensures privacy and security by training on local data
  - Allows for decentralized model training


