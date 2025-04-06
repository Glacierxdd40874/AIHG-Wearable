import torch
import pandas as pd
import psutil
import time
from datasets import load_dataset
from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context
from transformers import AutoModelForSequenceClassification
from fl.task import get_weights, load_data, set_weights, test, train

# Load and clean data
def clean_data_from_huggingface():
    # Load dataset
    dataset = load_dataset("sarthak-wiz01/nutrition_dataset")
    
    # Pick up data
    train_data = dataset['train']
    
    # Transformed to Pandas DataFrame
    df = pd.DataFrame(train_data)
    
    # delete suggestions
    df_cleaned = df.drop(columns=['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion'])
    
    return df_cleaned

# Load cleaned dataset
df_cleaned = clean_data_from_huggingface()
print(df_cleaned.head())

# Add noise
def add_noise(parameters, noise_scale=1.00):
    """Adding controllable noise to model parameters"""
    parameters_tensor = [torch.tensor(p) for p in parameters]  # Transform to PyTorch Tensor
    noise = [torch.randn_like(p) * noise_scale for p in parameters_tensor]  # Generate noise
    noisy_parameters = [p + n for p, n in zip(parameters_tensor, noise)]  # Add noise
    
    return [p.numpy() for p in noisy_parameters]  # 转回 numpy.ndarray

# Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs, noise_scale=1.00):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.noise_scale = noise_scale  # Directly define the noise scale
    
    def fit(self, parameters, config):
        """Training and adding noise"""
        # Setting model weights
        set_weights(self.net, parameters)

        # Recording time and memory usage
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss  # Record initial memory

        # Training the model
        train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)

        # Calculating memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_used = (final_memory - initial_memory) / (1024 * 1024)  # Convert to MB
        training_time = time.time() - start_time  # Calculate training time

        # Print training information
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Memory Used: {memory_used:.2f} MB")

        # Get the trained weights and add noise
        weights = get_weights(self.net)
        noisy_weights = add_noise(weights, self.noise_scale)  # Applying Noise

        return noisy_weights, len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluating the Model"""
        # Setting model weights
        set_weights(self.net, parameters)

        # Testing the Model
        loss, accuracy = test(self.net, self.testloader, self.device)

        # Display evaluation results
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        return float(loss), len(self.testloader), {"accuracy": accuracy}

# Client factory function
def client_fn(context: Context):
    """Creating a Flower Client"""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainloader, valloader = load_data(partition_id, num_partitions, model_name)

    # Loading the model
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    local_epochs = context.run_config["local-epochs"]
    
    # Define the noise value directly here, for example set it to 1.0
    noise_scale = 0.20  # Here we define the noise scale

    # Returns the client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, noise_scale).to_client()

# Create and start the Flower client
app = ClientApp(client_fn)
