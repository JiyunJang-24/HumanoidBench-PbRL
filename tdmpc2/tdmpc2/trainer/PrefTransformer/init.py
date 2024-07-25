import transformers
from .flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel
from .PrefTransformer import PrefTransformer
import os
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

def make_reward_model(cfg, obs_dim, action_dim):
    total_epochs = 30
    transformer = PrefTransformer.get_default_config()
    config = transformers.GPT2Config(
            **transformer
        )
    config.batch_size_RLHF = cfg.batch_size_RLHF
    config.min_delta = cfg.min_delta
    config.patience = cfg.patience
    config.n_epochs = cfg.n_epochs
    config.early_stop = cfg.early_stop
    config.eval_period = cfg.eval_period
    config.save_model = cfg.save_model
    config.batch_size_RLHF_eval = cfg.batch_size_RLHF_eval
    #data_size = pref_dataset["observations"].shape[0]
    data_size = int(cfg.batch_size * 0.75) // 2
    interval = int(data_size / config.batch_size_RLHF)

    config.warmup_steps = int(config.n_epochs * 0.1 * interval)
    config.total_steps = config.n_epochs * interval
    activations='relu'
    activation_final='none'
    
    trans = TransRewardModel(config=config, observation_dim=obs_dim, action_dim=action_dim, activation=activations, activation_final=activation_final)
    return PrefTransformer(config, trans)