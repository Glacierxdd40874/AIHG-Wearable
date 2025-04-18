import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
import os 

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize data cache
data = None

def load_model():
    model = Sequential([
    	Dense(128, input_dim=6, activation='relu'),  
    	Dense(64, activation='relu'),  
  	Dense(32, activation='relu'), 
    	Dense(1, activation='linear')  
	])

    # Modify the learning rate
    optimizer = Adam(learning_rate=0.001)  # Set the learning rate    
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

def load_data(partition_id, num_partitions, target_variable='sleep_quality'):
    global data

    if data is None:
        # Load the wellness data using absolute path
        data_path = r"E:\5703\Week7\wellness_data.csv"
        df = pd.read_csv(data_path)

        # Data Preprocessing: Remove unnecessary columns (timestamp and participant_id)
        df = df.drop(columns=["effective_time_frame", "participant_id", "date", "soreness_area"])

        # Select the target variable (sleep_quality or sleep_duration)
        if target_variable == 'sleep_quality':
            y = df["sleep_quality"].values  # Target: sleep quality
        elif target_variable == 'sleep_duration':
            y = df["sleep_duration"].values  # Target: sleep duration

        # Features are all columns except the target variable
        X = df.drop(columns=[target_variable]).values  # Features

        # Data normalization (optional)
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Store the data for future use
        data = (X_train, y_train, X_test, y_test)

    # Assign data according to partition_id and num_partitions
    x_train, y_train, x_test, y_test = data
    partition_size = len(x_train) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = (partition_id + 1) * partition_size if partition_id < num_partitions - 1 else len(x_train)

    return x_train[start_idx:end_idx], y_train[start_idx:end_idx], x_test, y_test
