import torch
import pandas as pd
import psutil
import time
from datasets import load_dataset
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import AutoModelForSequenceClassification
from fl.task import get_weights, load_data, set_weights, test, train
from phe import paillier
import json

# Loading the Paillier public key
def load_public_key():
    with open("public_key.json", "r") as f:
        public_key_data = json.load(f)
        n = int(public_key_data["n"], 16)
        g = int(public_key_data["g"], 16)
        return paillier.PaillierPublicKey(n, g)

# Load the dataset and clean it
def clean_data_from_huggingface():
    dataset = load_dataset("sarthak-wiz01/nutrition_dataset")
    train_data = dataset['train']
    df = pd.DataFrame(train_data)
    df_cleaned = df.drop(columns=['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion'])
    return df_cleaned

df_cleaned = clean_data_from_huggingface()
print(df_cleaned.head())

# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss  # load memory

        # train
        train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)

        final_memory = psutil.Process().memory_info().rss
        memory_used = (final_memory - initial_memory) / (1024 * 1024)  #  MB
        training_time = time.time() - start_time  # time

        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Memory Used: {memory_used:.2f} MB")

        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return float(loss), len(self.testloader), {"accuracy": accuracy}

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainloader, valloader = load_data(partition_id, num_partitions, model_name)

    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    local_epochs = context.run_config["local-epochs"]
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)
