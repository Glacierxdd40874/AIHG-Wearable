import torch
import pandas as pd
import psutil
import time
from datasets import load_dataset
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import AutoModelForSequenceClassification
from fl.task import get_weights, load_data, set_weights, test, train

# Load datasets and clean it
def clean_data_from_huggingface():
    # Load Data Set
    dataset = load_dataset("sarthak-wiz01/nutrition_dataset")
    
    # Extract training set
    train_data = dataset['train']
    
    # Transform to Pandas DataFrame
    df = pd.DataFrame(train_data)
    
    # Delete Suggestion
    df_cleaned = df.drop(columns=['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion'])
    
    return df_cleaned

# 加载清理后的数据
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
        # setweight
        set_weights(self.net, parameters)
        
        # time and memory
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss  # load memory
        
        # train
        train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)
        
        # caculate memory
        final_memory = psutil.Process().memory_info().rss
        memory_used = (final_memory - initial_memory) / (1024 * 1024)  #  MB
        training_time = time.time() - start_time  # time

        # result
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Memory Used: {memory_used:.2f} MB")

        # get weight after training
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # weight
        set_weights(self.net, parameters)
        
        # evaluation
        loss, accuracy = test(self.net, self.testloader, self.device)
        
        # 显示评估结果
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return float(loss), len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainloader, valloader = load_data(partition_id, num_partitions, model_name)

    # load model
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    local_epochs = context.run_config["local-epochs"]

    # return client
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
