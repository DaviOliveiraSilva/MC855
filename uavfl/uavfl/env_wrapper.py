import torch
from torchrl.envs import GymWrapper, TransformedEnv, Compose
from torchrl.envs.transforms import CatTensors
from uavfl.env import DroneEnv

def make_torchrl_env(map_size=50):
    """
    Cria um ambiente TorchRL a partir do DroneEnv (formato Gymnasium)
    Inclui transformação para achatar observações Dict em vetor plano
    """
    base_env = DroneEnv(map_size=map_size)
    env = GymWrapper(base_env)
    obs_keys = ["pos_x", "pos_y", "coverage", "battery", "wind_speed", "wind_angle", "altitude"]
    # Create a list of transforms
    transforms_list = [
        CatTensors(in_keys=obs_keys, out_key="observation", ),
    ]

    # Compose the transforms and apply them to the environment
    env = TransformedEnv(
        env,
        Compose(*transforms_list)
    )
    return env
