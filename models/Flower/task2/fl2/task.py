import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import Huber
import os
from keras.layers import BatchNormalization, Dropout

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Data cache
data = None

def load_model():
    
    model = Sequential([
    Dense(128, input_shape=(5,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  
])

    model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=Huber(delta=50.0),  
    metrics=["mae"]
    )
    return model

def load_data(partition_id, num_partitions):
    global data

    if data is None:
        data_path = r"E:\5703\Week11\t3.csv"
        df = pd.read_csv(data_path)


        df = df.dropna(subset=["Calories"])

        
        y = df["Calories"].values.reshape(-1, 1)  # shape (n_samples, 1)
        
        if 'BMI' in df.columns:
            df = df.drop(columns=['BMI'])

        
        X = df.drop(columns=["Calories"]).values  

        
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
        X = (X - mean) / std

        
        np.save(r"E:\5703\Week11\t3.npy", mean)
        np.save(r"E:\5703\Week11\t3.std", std)

        
        

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data = (X_train, y_train, X_test, y_test)

    # Partition training data for federated simulation
    X_train, y_train, X_test, y_test = data
    partition_size = len(X_train) // num_partitions
    start = partition_id * partition_size
    end = (partition_id + 1) * partition_size if partition_id < num_partitions - 1 else len(X_train)

    return X_train[start:end], y_train[start:end], X_test, y_test
