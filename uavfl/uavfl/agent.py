import torch
import torch.nn as nn
from torchrl.modules import MLP, ProbabilisticActor
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torch.distributions import Normal

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.mlp = MLP(in_features=state_dim, out_features=2 * action_dim, depth=2, num_cells=64)
        self.action_dim = action_dim # Store action_dim for convenience

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # Ensure 'observation' key exists in the input TensorDict
        if type(tensordict) is not TensorDict:
            tensordict = TensorDict({"observation": tensordict}, batch_size=[])
        obs = tensordict["observation"]
        out = self.mlp(obs)
        
        loc = out[..., :self.action_dim]
        # Use softplus to ensure scale is positive and add a small epsilon for stability
        scale = torch.nn.functional.softplus(out[..., self.action_dim:]) + 1e-5
        
        # Write 'loc' and 'scale' back into the input TensorDict
        tensordict["loc"] = loc
        tensordict["scale"] = scale
        return tensordict

class ValueNet(TensorDictModule):
    def __init__(self, state_dim):
        super().__init__(
            module=MLP(in_features=state_dim, out_features=1, depth=2, num_cells=64),
            in_keys=["observation"],
            out_keys=["state_value"]
        )


# Factory for the policy module
def make_policy_module(state_dim, action_dim):
    base_policy_net = PolicyNet(state_dim, action_dim)
    td_policy_module = TensorDictModule(
        base_policy_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    )
    
    # Now, use our custom SummingProbabilisticActor instead of the base ProbabilisticActor
    actor = ProbabilisticActor(
        module=td_policy_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=Normal,
        return_log_prob=True,
        # log_prob_key="sample_log_prob",
    )
    return actor # This actor is now the complete policy module