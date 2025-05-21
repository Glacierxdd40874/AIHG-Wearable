from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from fl2.task import load_model
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize the global model from your load_model function
    model = load_model()

    # Convert the model weights to parameters to be used by the server
    parameters = ndarrays_to_parameters(model.get_weights())

    # Define the federated learning strategy (FedProx)
    strategy = FedProx(
        fraction_fit=0.9,  # Increase fraction_fit to ensure more clients participate
        fraction_evaluate=1.0,  # Ensure all clients are evaluated
        min_available_clients=3,  # Ensure at least 3 clients are available to participate
        initial_parameters=parameters,  # Initial global model parameters
        proximal_mu=0.1  # FedProx regularization parameter, typically small (e.g., 0.1 or 0.01)
    )

    # Define the configuration for the server (number of rounds)
    config = ServerConfig(num_rounds=num_rounds)

    # Return the ServerApp components (strategy and config)
    return ServerAppComponents(strategy=strategy, config=config)

# Create and start the ServerApp
app = ServerApp(server_fn=server_fn)

# Run the Flower server
if __name__ == "__main__":
    app.run()
