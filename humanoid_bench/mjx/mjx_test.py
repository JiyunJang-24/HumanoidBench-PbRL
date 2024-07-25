import os
import mujoco.viewer
import numpy as np
import time
from .flax_to_torch import TorchModel, TorchPolicy
from humanoid_bench.mjx.envs.cpu_env import HumanoidNumpyEnv
import tqdm
from .video_utils import save_numpy_as_video, make_grid_video_from_numpy

def main(args):
    if args.with_full_model:
        env = HumanoidNumpyEnv('./humanoid_bench/assets/mjx/scene_test_mesh_collisions_hands_two_targets_pos.xml', task=args.task)
    else:
        env = HumanoidNumpyEnv('./humanoid_bench/assets/mjx/scene_mjx_feet_collisions_two_targets_pos.xml', task=args.task)
    
    state = env.reset()
    print("State:", state)

    if args.task == 'reach':
        torch_model = TorchModel(55, 19)
    elif args.task == 'reach_two_hands':
        torch_model = TorchModel(61, 19)
    torch_policy = TorchPolicy(torch_model)

    model_name = "torch_model.pt"
    mean_name = "mean.npy"
    var_name = "var.npy"

    if args.step is not None:
        model_name = model_name.split(".")[0] + f"_{args.step}." + model_name.split(".")[1]
        mean_name = mean_name.split(".")[0] + f"_{args.step}." + mean_name.split(".")[1]
        var_name = var_name.split(".")[0] + f"_{args.step}." + var_name.split(".")[1]

    torch_policy.load(os.path.join(args.folder, model_name), mean=os.path.join(args.folder, mean_name), var=os.path.join(args.folder, var_name))
    
    m, d = env.model, env.data

    if args.render:
        rollout_number = 1
        all_rewards = []
        all_videos = []

        renderer = mujoco.Renderer(m, height=480, width=480)
        for _ in tqdm.tqdm(range(rollout_number)):
            state = env.reset()
            i = 0
            reward = 0
            video = []
            while True:
                action = torch_policy(state)
                state, r, done, _ = env.step(action)
                reward += r
                i += 1
                renderer.update_scene(d, camera='cam_default')
                frame = renderer.render()
                video.append(frame)
                if done or i > 1000:
                    break
            all_rewards.append(reward)
            all_videos.append(np.array(video))
        make_grid_video_from_numpy(all_videos, 8, output_name=os.path.join(args.folder, "evaluation.mp4"), **{'fps': 50})
        print("Rewards:", all_rewards)
    else:
        # Rollout with mujoco viewer
        # NOTE: reset not implemented, need to manually relaunch the script
        with mujoco.viewer.launch_passive(m, d) as viewer:
            state = env.reset()
            print("State:", state)
            viewer.sync()
            time.sleep(0.02)
            while viewer.is_running():
               
                action = torch_policy(state)
                state, _, _, _ = env.step(action)
               
                viewer.sync()
                time.sleep(0.02)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./data/reach_one_hand/')
    parser.add_argument('--task', type=str, default='reach')
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--with_full_model', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
