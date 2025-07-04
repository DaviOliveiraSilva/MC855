# federated_ppo_drones.py
import torch
import flwr as fl
from agent import PPOAgent
from env import DroneEnv

STATE_DIM = 8
ACTION_DIM = 2
LOCAL_EPISODES = 5
LOCAL_UPDATES = 3

class FederatedDroneClient(fl.client.NumPyClient):
    def __init__(self):
        self.agent = PPOAgent(STATE_DIM, ACTION_DIM)
        self.env = DroneEnv()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.agent.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(self.agent.model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.agent.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trajectories = []

        for _ in range(LOCAL_EPISODES):
            state = self.env.reset()
            done = False
            while not done:
                action, log_prob = self.agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                trajectories.append((state, action, reward, done, log_prob))
                state = next_state

        for _ in range(LOCAL_UPDATES):
            self.agent.update(trajectories)

        return self.get_parameters({}), len(trajectories), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        total_reward = 0
        episodes = 5

        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action, _ = self.agent.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

        avg_reward = total_reward / episodes
        return -avg_reward, episodes, {"reward": avg_reward}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FederatedDroneClient())