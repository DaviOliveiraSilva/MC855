"""uavFL: A Flower / PyTorch app."""

import torch
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tensordict import TensorDict

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from uavfl.task import get_weights, set_weights
from uavfl.env_wrapper import make_torchrl_env
from uavfl.agent import ValueNet, make_policy_module

STATE_DIM = 7
ACTION_DIM = 2
LOCAL_UPDATES = 5
MAP_SIZE = 50

class FlowerClient(NumPyClient):
    def __init__(self):
        # Determine the device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}") # For debugging

        self.policy = make_policy_module(STATE_DIM, ACTION_DIM).to(self.device)
        self.value = ValueNet(STATE_DIM).to(self.device)

        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.optim_value = torch.optim.Adam(self.value.parameters(), lr=1e-3)

        self.loss_module = ClipPPOLoss(
            actor=self.policy,
            critic=self.value,
            clip_epsilon=0.2,
            entropy_bonus=True,
            entropy_coef=0.01,
        )

    def fit(self, parameters, config):
        set_weights(self.policy, self.value, parameters)

        train_env = make_torchrl_env(MAP_SIZE).to(self.device)
        collector = SyncDataCollector(
            train_env,
            self.policy,
            frames_per_batch=1000,
            total_frames=1000 * LOCAL_UPDATES,
            device=self.device,
        )

        # Training loop
        for i, batch in enumerate(collector):
            # Move the batch to the correct device
            batch = batch.to(self.device)

            if "next" in batch.keys():
                if "reward" in batch["next"].keys():
                    print("Found 'reward' key nested under 'next'. Moving it to top level.")
                    batch["reward"] = batch["next"]["reward"]
                # del batch["next"]["reward"] 
            else:
                print("ERROR: 'reward' key not found in collected batch at top level or under 'next'!")
                # Print keys of 'next' TensorDict if it exists
                print(f"Keys in 'next' tensordict: {batch['next'].keys() if 'next' in batch.keys() else 'No next tensordict'}")
                raise KeyError("Reward key missing from collected batch after attempt to fix.")
            
            with torch.no_grad():
                self.value(batch) # Populates batch['state_value']
                
                # More robust check for batch['next'] and its contents
                if "next" in batch.keys():
                    if isinstance(batch["next"], TensorDict):
                        # Only compute next_state_value if 'observation' is present in 'next'
                        if "observation" in batch["next"].keys():
                            self.value(batch["next"]) # Populates batch['next']['state_value']
                            print("ValueNet executed on batch['next']: 'state_value' should now be in batch['next'].")
                        else:
                            print("WARNING: batch['next'] is a TensorDict but 'observation' key is missing. Cannot populate next_state_value.")
                    else:
                        print(f"WARNING: batch['next'] exists but is not a TensorDict (type: {type(batch['next'])}). Cannot populate next_state_value.")
                else:
                    print("WARNING: 'next' key not found in batch. Cannot populate next_state_value.")
            if "sample_log_prob" in batch.keys():
                current_log_prob = batch["sample_log_prob"]
                original_shape = current_log_prob.shape
                print(f"sample_log_prob shape before manual reshape: {original_shape}")

                # If shape is (Batch, ACTION_DIM, 1), sum over ACTION_DIM and squeeze last dim
                if current_log_prob.ndim == 3 and current_log_prob.shape[1] == ACTION_DIM and current_log_prob.shape[2] == 1:
                    print(f"Warning: Reshaping collected sample_log_prob from {original_shape} to (Batch, 1) by squeezing.")
                    batch["sample_log_prob"] = current_log_prob.sum(dim=1, keepdim=True).squeeze(-1)
                # If shape is (Batch, ACTION_DIM), sum over ACTION_DIM
                elif current_log_prob.ndim == 2 and current_log_prob.shape[1] == ACTION_DIM:
                    print(f"Warning: Reshaping collected sample_log_prob from {original_shape} to (Batch, 1) by summing.")
                    batch["sample_log_prob"] = current_log_prob.sum(dim=1, keepdim=True)
                # If shape is (Batch,), unsqueeze to (Batch, 1)
                elif current_log_prob.ndim == 1:
                    print(f"Warning: Reshaping collected sample_log_prob from {original_shape} to (Batch, 1) by unsqueezing.")
                    batch["sample_log_prob"] = current_log_prob.unsqueeze(-1)

                print(f"sample_log_prob shape after manual reshape: {batch['sample_log_prob'].shape}")

            loss_vals = self.loss_module(batch)
            print("Loss module forward pass completed.")
            loss = loss_vals["loss_total"]

            # Optimize policy and value networks
            self.optim_policy.zero_grad()
            self.optim_value.zero_grad()
            loss.backward()
            self.optim_policy.step()
            self.optim_value.step()

            # Break if enough local updates have been performed
            if i >= LOCAL_UPDATES - 1:
                break 

        collector.shutdown() # Important: release environment resources

        # Return updated weights, number of examples, and metrics
        # The number of examples should reflect the actual data used for training.
        # Here, it's the total frames collected.
        return get_weights(self.policy, self.value), collector.total_frames, {}

    def evaluate(self, parameters, config):
        set_weights(self.policy, self.value, parameters)
        
        eval_env = make_torchrl_env().to(self.device)
        
        total_reward = 0.0

        eval_tensordict = eval_env.reset()

       # Explicit Batching
        if eval_tensordict.batch_size == torch.Size([]):
            eval_tensordict = eval_tensordict.unsqueeze(0)
        
        with torch.no_grad():
            while True:
                with set_exploration_type(ExplorationType.MEAN):
                    # Actor expects a TensorDict as input
                    action_tensordict = self.policy(eval_tensordict.to(self.device))
                    action = action_tensordict["action"]

                    next_eval_tensordict = eval_env.step(action_tensordict)
                    # Ensure the stepped tensordict is batched for the next loop iteration
                    if next_eval_tensordict.batch_size == torch.Size([]):
                        next_eval_tensordict = next_eval_tensordict.unsqueeze(0)

                    # Check if 'reward' is present in the stepped tensordict before accessing
                    if "reward" not in next_eval_tensordict.keys():
                        print("ERROR: 'reward' key not found in tensordict after env.step() in evaluate loop!")
                        print(f"Keys available: {next_eval_tensordict.keys()}")
                        if "next" in next_eval_tensordict.keys() and "reward" in next_eval_tensordict["next"].keys():
                            print("Found 'reward' key nested under 'next' in evaluation step. Using it.")
                            # Squeeze to remove batch dimension if present before item()
                            reward = next_eval_tensordict["next"]["reward"].squeeze().item()
                        else:
                            raise KeyError("Reward key missing from evaluation step tensordict.")
                    else:
                        # Squeeze to remove batch dimension if present before item()
                        reward = next_eval_tensordict["reward"].squeeze().item()
                    
                    # Squeeze 'done' to remove batch dimension if present before item()
                    done = next_eval_tensordict["done"].squeeze().item()

                    total_reward += reward
                    
                    eval_tensordict = next_eval_tensordict # Update eval_tensordict for the next loop

                    if done:
                        break
        
        eval_env.close() # Close the evaluation environment

        # Return negative reward as loss (common for RL evaluation), number of examples, and metrics
        return float(-total_reward), 1, {"reward": float(total_reward)}


def client_fn(context: Context):
    return FlowerClient().to_client()
    
app = ClientApp(
    client_fn,
)