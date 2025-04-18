import time
import psutil
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from fl2.task import load_data, load_model

class FlowerClient(NumPyClient):
    def __init__(self, model, data, epochs, batch_size, verbose):
        self.model = model
        # 数据不再包含时间和参与者ID字段
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        # Set model weights
        self.model.set_weights(parameters)

        # Start time and memory tracking
        start_time = time.time()
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # Memory in MB

        # Train the model
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        # End time and memory tracking
        end_time = time.time()
        memory_after = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        training_time = end_time - start_time  # Training time in seconds
        memory_usage = memory_after - memory_before  # Memory used during training in MB

        # Print the results for each round
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Memory used: {memory_usage:.2f} MB")
        
        # Print training history (loss and accuracy at each epoch)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}: loss = {history.history['loss'][epoch]:.4f}, mae = {history.history['mae'][epoch]:.4f}")

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        # Set model weights
        self.model.set_weights(parameters)

        # Start time and memory tracking
        start_time = time.time()
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # Memory in MB

        # Evaluate the model
        loss, mae = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        # End time and memory tracking
        end_time = time.time()
        memory_after = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        evaluation_time = end_time - start_time  # Evaluation time in seconds
        memory_usage = memory_after - memory_before  # Memory used during evaluation in MB

        # Print the results
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")
        print(f"Memory used: {memory_usage:.2f} MB")
        print(f"Loss: {loss:.4f}, MAE: {mae:.4f}")

        return loss, len(self.x_test), {"mae": mae}


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    # Get partition info
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    
    # Ensure data doesn't include time and participant_id fields
    x_train, y_train, x_test, y_test = data
    data = (x_train, y_train, x_test, y_test)

    # Get configuration for epochs, batch size, etc.
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return FlowerClient instance
    return FlowerClient(net, data, epochs, batch_size, verbose).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
