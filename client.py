import flwr as fl
import torch
import torch.optim as optim
from agent import PolicyNetwork
from env import DroneEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM = 7
ACTION_DIM = 2

class DroneClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = PolicyNetwork(STATE_DIM, ACTION_DIM).to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        env = DroneEnv()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for episode in range(5):  # Local episodes
            state = torch.tensor(env.reset(), dtype=torch.float32).to(DEVICE)
            done = False
            while not done:
                action, log_prob = self.model.act(state)
                next_state, reward, done, _ = env.step(action.cpu().detach().numpy())
                next_state = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)

                loss = -log_prob * reward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state

        return self.get_parameters({}), 1, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        env = DroneEnv()

        total_reward = 0.0
        for _ in range(5):
            state = torch.tensor(env.reset(), dtype=torch.float32).to(DEVICE)
            done = False
            while not done:
                with torch.no_grad():
                    action, _ = self.model.act(state)
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                total_reward += reward
                state = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)

        avg_reward = total_reward / 5
        return -avg_reward, 1, {"reward": avg_reward}

fl.client.start_numpy_client(server_address="localhost:8080", client=DroneClient())
