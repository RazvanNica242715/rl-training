import time

import gymnasium as gym
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback

import ot2_env  # needed to trigger registration
import wandb

# run = wandb.init(project="OT2ENV-v0", sync_tensorboard=True)
# wandb_callback = WandbCallback(
#     model_save_freq=100_000,
#     model_save_path=f"models/{run.id}",
#     verbose=1,
# )

# Training
max_steps = 2000
env = gym.make("OT2ENV-v0", max_steps=max_steps, render_mode="none")

# print("Starting training...")
# model = PPO("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", verbose=0, device="cpu")
# model.learn(
#     total_timesteps=2_000_000,
#     progress_bar=True,
#     callback=wandb_callback,
# )
# print("Done.")
# env.close()
# wandb.finish()

# # Save the model
# # model.save("ppo_ot2_test")

# Load the model from checkpoint
model = PPO.load("models/awbno1a1/model.zip", device="cpu")

# Testing
env = gym.make("OT2ENV-v0", max_steps=1000, render_mode="human")
obs, info = env.reset()

for i in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    print(f"Step {i}: Reward={reward:.3f}, Distance={info['distance_to_target']:.6f}m")

    if done:
        print(f"Episode finished! Final distance: {info['distance_to_target']:.6f}m")
        time.sleep(2)
        obs, info = env.reset()

env.close()
