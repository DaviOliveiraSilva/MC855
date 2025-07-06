"""uavFL: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torchrl.modules import MLP, SafeModule, NormalParamWrapper
from uavfl.task import get_weights
from uavfl.agent import ValueNet, make_policy_module

STATE_DIM = 7
ACTION_DIM = 2

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    policy = make_policy_module(STATE_DIM, ACTION_DIM)
    value = ValueNet(STATE_DIM)

    # Initialize model parameters
    ndarrays = get_weights(policy, value)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
