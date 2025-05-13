import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Data cache
data = None

def load_model():
    model = Sequential([
        Dense(128, input_shape=(5,), activation='relu'),  
        Dense(64, activation='relu'),  
        Dense(32, activation='relu'), 
        Dense(2, activation='linear')  # Predicting [sleep_duration_h, sleep_quality]
    ])

    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

def load_data(partition_id, num_partitions):
    global data

    if data is None:
        data_path = r"E:\5703\Week7\wellness_data_enhanced.csv"
        df = pd.read_csv(data_path)

        # Drop irrelevant columns
        df = df.drop(columns=["effective_time_frame", "participant_id", "date", "soreness_area"])

        # Define features and targets
        y1 = df["sleep_duration_h"].values.reshape(-1, 1)  # 0-12
        y2 = df["sleep_quality"].values.reshape(-1, 1)   # 0-5
        y = np.concatenate([y1, y2], axis=1)             # shape (n_samples, 2)

        # Keep only the required features
        X = df.drop(columns=["sleep_duration_h", "sleep_quality"]).values  # fatigue, mood, readiness, soreness, stress

        # Normalize input features
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Save mean and std
        np.save(r"E:\5703\Week7\mean.npy", mean)
        np.save(r"E:\5703\Week7\std.npy", std)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data = (X_train, y_train, X_test, y_test)

    # Partition training data for federated simulation
    X_train, y_train, X_test, y_test = data
    partition_size = len(X_train) // num_partitions
    start = partition_id * partition_size
    end = (partition_id + 1) * partition_size if partition_id < num_partitions - 1 else len(X_train)

    return X_train[start:end], y_train[start:end], X_test, y_test
