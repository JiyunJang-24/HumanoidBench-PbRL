from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
import cv2
from tdmpc2.trainer.base import Trainer_RLHF
#from tdmpc2.trainer.PrefTransformer import PrefTransformer_Trainer
import random
import os
from tqdm import tqdm, trange
import imageio
from IPython.display import display, Video


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

class OnlineTrainer(Trainer_RLHF):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                # self.logger.video.save(self._step)
                self.logger.video.save(self._step, key='results/video')
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def train(self):
        
        ## DW
        # pref_trainer = PrefTransformer_Trainer(self.reward_model)
        reward_step = 0
        
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, True
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=info["success"],
                    )
                    train_metrics.update(self.common_metrics())

                    results_metrics = {'return': train_metrics['episode_reward'],
                                       'episode_length': len(self._tds[1:]),
                                       'success': train_metrics['episode_success'],
                                       'success_subtasks': info['success_subtasks'],
                                       'step': self._step,}
                
                    self.logger.log(train_metrics, "train")
                    self.logger.log(results_metrics, "results")
                    #TODO _tds에서 self.reward_model의 self._tds의 reward 추가
                    #print(len(self._tds))
                    if self._step >= 200000:
                        batch = {}
                        batch['observations'] = []
                        batch['actions'] = []
                        batch['timestep'] = []
                        batch['attn_mask'] = []

                        for t in range(len(self._tds)-1):
                            batch['observations'].append(self._tds[t]["obs"].numpy())  # 텐서를 numpy 배열로 변환하여 추가
                            batch['actions'].append(self._tds[t+1]["action"].numpy())    # 텐서를 numpy 배열로 변환하여 추가
                            batch['timestep'].append(t)
                            batch['attn_mask'].append(1)  # 적절한 attention mask 값 설정 (예: 1)

                        # 리스트를 numpy 배열로 변환
                        batch['observations'] = np.expand_dims(np.squeeze(np.array(batch['observations'], dtype=np.float32)), axis=0) #B * seq_len * dim
                        batch['actions'] = np.expand_dims(np.squeeze(np.array(batch['actions'], dtype=np.float32)), axis=0)
                        batch['timestep'] = np.expand_dims(np.squeeze(np.array(batch['timestep'], dtype=np.int64)), axis=0)
                        batch['attn_mask'] = np.expand_dims(np.squeeze(np.array(batch['attn_mask'], dtype=np.int64)), axis=0)
                        pref_rewards, _ = self.reward_model.get_reward(batch)
                        #print(batch['observations'], batch['actions'])
                        #print(pref_rewards)
                        for t in range(len(self._tds)-1):
                            reward_np = np.array(pref_rewards[0, 0, t])
                            reward_tensor = torch.tensor(reward_np, dtype=self._tds[t+1]['reward'].dtype)
                            
                            self._tds[t+1]['reward'] += reward_tensor
                        
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))
                    #print(self._ep_idx, torch.cat(self._tds))
                    #58 TensorDict(fields={action : Tensor(shape=torch.Size([22,19])) obs : Tensor(shape=torch.Size([22,64])), reward: Tensor(shape=torch.Size[22])})

                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]
                #print(self._tds, self.to_td(obs))
                #import pdb; pdb.set_trace()

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, truncated, info = self.env.step(action)
            #print(self.env.get_state())
            #print(dir(self.env))
            #print(dir(self.env.env.env))
            #print(info)
            done = done or truncated
            self._tds.append(self.to_td(obs, action, reward))
            
            if self._step != 0:
                if self._step % 200000 == 0:
                    obs_RLHF, action_RLHF, reward_RLHF, episode_RLHF, task_RLHF = self.buffer.sample_for_RLHF()
                    #print(episode[0][0], episode[1][0])
                    #print(obs.size())
                    save_dir = './video'
                    env_name = self.cfg.task
                    data_pair = self.random_sampling_(action_RLHF)
                    num_query = action_RLHF.size(1)//2
                    with_out_video = False
                    
                    self.create_video_sample(obs=obs_RLHF, action=action_RLHF, reward=reward_RLHF, task=task_RLHF, num_query=num_query, data_pair=data_pair,
                                             save_dir=save_dir, env_name=env_name, width=500, height=500, with_out_video=with_out_video)
                    #with_out_video for Debugging
                    save_label = self.create_human_label(save_dir=save_dir, env_name=env_name, num_query=num_query, start_idx=None, width=500, height=500, with_out_video=with_out_video)
                    #TODO reward_model와 train 코드 작성해야함
                    ## DW
                    logger_info = {"reward_step" : reward_step, "global_step" : self._step}
                    reward_step += 1
                    
                    self.reward_model.train_RLHF(obs=obs_RLHF, action=action_RLHF, reward=reward_RLHF, episode=episode_RLHF, task=task_RLHF, data_pair=data_pair,
                                             save_dir=save_dir, env_name=env_name, num_query=num_query, 
                                                 query_len = action_RLHF.size()[0], save_label=save_label, logger=self.logger, logger_info=logger_info)
                    print("Reward model is trained successfully in step ", self._step)
                    # print(reward_model_metric)
            
            
            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1
                    
                for _ in range(num_updates):
                    
                    # self.logger.log(reward_model_metric, "reward_model")
                    #print(obs.size(), action.size(), reward.size(), episode.size())
                    #torch.Size([4, 256, 64]), torch.Size([3, 256, 19]), torch.Size([3, 256, 1]) 한 에피소드에서 3개의 step을 가져왔다는 뜻, 즉, episode[0][0]와 episode[1][0] 은 같은 episode은데 step이 0 과 1인것!
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1

        self.logger.finish(self.agent)
        
        
    def random_sampling_(self, action):
        length = action.size()[1]
        
        indices = list(range(length))
        random_pairs = []

        # Randomly shuffle the indices
        random.shuffle(indices)

        # Create pairs from the shuffled indices
        for i in range(0, length, 2):
            random_pairs.append((indices[i], indices[i+1]))

        return random_pairs


    def create_video_sample(self, obs, action, reward, task, num_query, data_pair, save_dir, env_name, width=500, height=500, with_out_video=False):
        # Determine the video save path
        video_path = os.path.join(save_dir, env_name, f"iter_idx{self._step}")
        os.makedirs(os.path.join(video_path), exist_ok=True)
        
        if with_out_video==True:
            return False
        
        #num_query = action.size()[1]//2
        query_len = obs.size()[0]
        qpos_len = self.env.model.nq
        qvel_len = self.env.model.nv
        # if len(data_pair) < num_query :
        #     print("data_pair is too small!")
        for seg_idx in trange(num_query):

            # start_1, start_2 = (
            #     batch["start_indices"][seg_idx],
            #     batch["start_indices_2"][seg_idx],
            # )
            start_1 = data_pair[seg_idx][0]
            start_2 = data_pair[seg_idx][1]
            frames = []
            frames_2 = []

            #start_indices = range(start_1, start_1 + query_len)
            #start_indices_2 = range(start_2, start_2 + query_len)

            self.env.reset()

            camera_name = "track"

            for t in trange(query_len, leave=False):
                state = obs[t][start_1].cpu().numpy()  # Assuming obs is a tensor
                qpos = state[:qpos_len]
                qvel = state[qpos_len:qpos_len+qvel_len]
                self.env.set_state(qpos, qvel)
                #self.env.set_state(np.array(obs[t][start_1][:qpos_len].cpu()), np.array(obs[t][start_1][qpos_len:].cpu()))
                #curr_frame = gym_env.sim.render(width=width, height=height, mode="offscreen", camera_name=camera_name)
                curr_frame = self.env.render()
                frames.append((curr_frame))
            
            self.env.reset()
            
            for t in trange(query_len, leave=False):
                state = obs[t][start_2].cpu().numpy()  # Assuming obs is a tensor
                qpos = state[:qpos_len]
                qvel = state[qpos_len:qpos_len+qvel_len]
                self.env.set_state(qpos, qvel)
                #self.env.set_state(obs[t][start_2][:qpos_len], obs[t][start_2][qpos_len:])
                
                curr_frame = self.env.render()
                frames_2.append((curr_frame))

            video = np.concatenate((np.array(frames), np.array(frames_2)), axis=2)

            writer = imageio.get_writer(os.path.join(save_dir, env_name, f"iter_idx{self._step}", f"./idx{seg_idx}.mp4"), fps=30)
            for frame in tqdm(video, leave=False):
                writer.append_data(frame)
            writer.close()
        


    
    def create_human_label(self, save_dir, env_name, num_query, start_idx=None, width=2000, height=1500, with_out_video=False):
        video_path = os.path.join(save_dir, env_name)
        os.makedirs(os.path.join(video_path, f"iter_idx{self._step}", "label"), exist_ok=True)
        print("START!")
        if start_idx:
            assert start_idx > 0, "you must input with video number (1, 2, 3, ...)"
            interval = range(start_idx - 1, num_query)
        else:
            interval = range(num_query)
        label2 = []
        
        
        if with_out_video == True:
            for i in interval:
                label2.append(get_label(random.randint(1, 3)))
            return np.array(label2)
        
        
        for i in interval:
            label = False
            while not label:
                
                print(f"\nVideo {i + 1}")
                video_file = os.path.join(video_path, f"iter_idx{self._step}", f"./idx{i}.mp4")
                #display(Video(video_file, width=width, height=height, html_attributes="loop autoplay"))
                cap = cv2.VideoCapture(video_file)
                print(f"[{i + 1}/{num_query}] Put Preference (1 (left), 2 (right), 3 (equal)):  ")

                if not cap.isOpened():
                    print(f"Error opening video file {video_file}")
                    break
                
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        # If the video reaches the end, reset the capture to the beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    cv2.imshow(f"Video {i + 1}", frame)
                    
                    # Resize window
                    cv2.resizeWindow(f"Video {i + 1}", width, height)
                    
                    # Press 'Q' on keyboard to exit viewing
                    key = cv2.waitKey(100)  # Increase the delay to slow down the video
                    if key == ord('1'):
                        reward = 1
                        break
                    elif key == ord('2'):
                        reward = 2
                        break
                    elif key == ord('3'):
                        reward = 3
                        break
                    
                cap.release()
                cv2.destroyAllWindows()
                label = get_label(reward)
                #print(reward)
                if label:
                    with open(os.path.join(video_path, f"iter_idx{self._step}", "label", f"label_{i}.txt"), "w") as f:
                        f.write(str(reward))
                    label2.append(label)
        return np.array(label2)
