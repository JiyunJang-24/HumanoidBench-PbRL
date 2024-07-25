import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from IPython.display import Video
from tdmpc2.common.buffer import Buffer
from tdmpc2.trainer.base import Trainer

def get_label(ans):
    try:
        ans = int(ans)
    except:
        print("Wrong Input")
        return False
    if ans not in [1,2,3]:
        print("Invalid option.")
        return False
    if ans == 1:
        return [1, 0]
    elif ans == 2:
        return [0, 1]
    else:
        return [0.5, 0.5]


class OfflineTrainer(Trainer):
    """Trainer class for multi-task offline TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time()

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        results = dict()
        for task_idx in tqdm(range(len(self.cfg.tasks)), desc="Evaluating"):
            ep_rewards, ep_successes = [], []
            for _ in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset(task_idx)[0], False, 0, 0
                while not done:
                    action = self.agent.act(
                        obs, t0=t == 0, eval_mode=True, task=task_idx
                    )
                    obs, reward, done, truncated, info = self.env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    t += 1
                ep_rewards.append(ep_reward)
                ep_successes.append(info["success"])
            results.update(
                {
                    f"episode_reward+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_rewards
                    ),
                    f"episode_success+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_successes
                    ),
                }
            )
        return results

    def train(self):
        """Train a TD-MPC2 agent."""
        assert self.cfg.multitask and self.cfg.task in {
            "mt30",
            "mt80",
        }, "Offline training only supports multitask training with mt30 or mt80 task sets."

        # Load data
        assert self.cfg.task in self.cfg.data_dir, (
            f"Expected data directory {self.cfg.data_dir} to contain {self.cfg.task}, "
            f"please double-check your config."
        )
        fp = Path(os.path.join(self.cfg.data_dir, "*.pt"))
        fps = sorted(glob(str(fp)))
        assert len(fps) > 0, f"No data found at {fp}"
        print(f"Found {len(fps)} files in {fp}")

        # Create buffer for sampling
        _cfg = deepcopy(self.cfg)
        _cfg.episode_length = 101 if self.cfg.task == "mt80" else 501
        _cfg.buffer_size = 550_450_000 if self.cfg.task == "mt80" else 345_690_000
        _cfg.steps = _cfg.buffer_size
        self.buffer = Buffer(_cfg)
        for fp in tqdm(fps, desc="Loading data"):
            td = torch.load(fp)
            assert td.shape[1] == _cfg.episode_length, (
                f"Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, "
                f"please double-check your config."
            )
            for i in range(len(td)):
                self.buffer.add(td[i])
        assert (
            self.buffer.num_eps == self.buffer.capacity
        ), f"Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes."
        
        self.create_video_sample(index=self.uniform_sampling_buffer(), save_dir=save_dir, env_name=env_name)
        self.create_human_label(save_dir=save_dir, env_name=env_name)
        #PreferenceTransformer()
        
        
        print(f"Training agent for {self.cfg.steps} iterations...")
        metrics = {}
        for i in range(self.cfg.steps):
            # Update agent
            train_metrics = self.agent.update(self.buffer)

            # Evaluate agent periodically
            if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
                metrics = {
                    "iteration": i,
                    "total_time": time() - self._start_time,
                }
                metrics.update(train_metrics)
                if i % self.cfg.eval_freq == 0:
                    metrics.update(self.eval())
                    self.logger.pprint_multitask(metrics, self.cfg)
                    if i > 0:
                        self.logger.save_agent(self.agent, identifier=f"{i}")
                self.logger.log(metrics, "pretrain")

        self.logger.finish(self.agent)
    
    def uniform_sampling_buffer(self):
        buffer = self.buffer
        print(len(buffer))
        import pdb; pdb.set_trace()
        return random.sample(range(0, len(buffer)), len(buffer//100))
    
    def create_video_sample(self, index, save_dir, env_name):
        return 1
    
    def create_human_label(self, save_dir, env_name, num_query=1000, start_idx=None, width=1000, height=500):
        video_path = os.path.join(save_dir, env_name)
        os.makedirs(os.path.join(video_path, "label"), exist_ok=True)
        print("START!")
        if start_idx:
            assert start_idx > 0, "you must input with video number (1, 2, 3, ...)"
            interval = range(start_idx - 1, num_query)
        else:
            interval = range(num_query)
            
        for i in interval:
            label = False
            while not label:
                print(f"\nVideo {i + 1}")
                video_file = os.path.join(video_path, f"idx{i}.mp4")
                display(Video(video_file, width=width, height=height, html_attributes="loop autoplay"))
                reward = input(f"[{i + 1}/{num_query}] Put Preference (1 (left), 2 (right), 3 (equal)):  ").strip()
                label = get_label(reward)
                if label:
                    with open(os.path.join(video_path, "label", f"label_{i}.txt"), "w") as f:
                        f.write(reward)
