from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from fl2.task import load_model
import os



class SaveModelStrategy(FedProx):
    def __init__(self, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds

    def aggregate_fit(self, server_round, results, failures):
        # Extract Indicators
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is None:
            return None

        parameters, metrics = aggregated_result

        if server_round == self.num_rounds:
            model = load_model()
            ndarrays = parameters_to_ndarrays(parameters)
            model.set_weights(ndarrays)

            # SaveModel
            save_path = r"E:\5703\Week7\final_model.h5"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            print(f"\n✅ Final model saved to {save_path}")

        return parameters, metrics

def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    model = load_model()
    parameters = ndarrays_to_parameters(model.get_weights())
    
    strategy = SaveModelStrategy(
        num_rounds=num_rounds,
        fraction_fit=0.9,
        fraction_evaluate=1.0,
        min_available_clients=3,
        initial_parameters=parameters,
        proximal_mu=0.1
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    app.run()
