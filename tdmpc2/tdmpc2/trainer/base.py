from .PrefTransformer.init import make_reward_model

class Trainer:
    """Base trainer class for TD-MPC2."""

    def __init__(self, cfg, env, agent, buffer, logger):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        print("Architecture:", self.agent.model)
        print("Learnable parameters: {:,}".format(self.agent.model.total_params))

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        raise NotImplementedError

    def train(self):
        """Train a TD-MPC2 agent."""
        raise NotImplementedError

class Trainer_RLHF:
    """Base trainer class for TD-MPC2."""

    def __init__(self, cfg, env, agent, buffer, reward_model, logger):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        if reward_model == "PrefTransformer":
            obs = self.env.reset()[0]
            action = self.env.action_space.sample()            
            state_dim = obs.shape[0]
            action_dim = action.shape[0]
            
            self.reward_model = make_reward_model(self.cfg, obs_dim=state_dim, action_dim=action_dim)
        print("Architecture:", self.agent.model)
        
        print("Learnable parameters: {:,}".format(self.agent.model.total_params))
        print("Reward model Architecture:", self.reward_model)
        
    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        raise NotImplementedError

    def train(self):
        """Train a TD-MPC2 agent."""
        raise NotImplementedError
