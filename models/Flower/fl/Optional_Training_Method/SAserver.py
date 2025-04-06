from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from transformers import AutoModelForSequenceClassification
from phe import paillier
from fl.task import get_weights

# Generate Paillier public and private keys
def generate_paillier_keys():
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

# Save the public key as a file
def save_public_key(public_key):
    with open("server_public_key.txt", "w") as pub_file:
        pub_file.write(f"n: {public_key.n}\ng: {public_key.g}\n")

# Defining server-side functionality
def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    model_name = context.run_config["model-name"]
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    # Generate a Paillier public key and save it
    public_key, _ = generate_paillier_keys()
    save_public_key(public_key)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
