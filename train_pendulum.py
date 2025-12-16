import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import gymnasium as gym
import os
import warnings
from clearml import Task

task = Task.init(project_name='Pendulum-v1/test',
                 task_name='Experiment1_67')

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--total_timesteps", type=int, default=100000)
parser.add_argument("--num_iterations", type=int, default=10)

args = parser.parse_args()

env = gym.make('Pendulum-v1', g=9.81)

run = wandb.init(
    project="sb3_pendulum_demo",
    sync_tensorboard=True,
    config={
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "total_timesteps": args.total_timesteps,
        "num_iterations": args.num_iterations,
    },
    settings=wandb.Settings(_disable_stats=True)  # Reduce logging overhead
)

model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=args.learning_rate, 
    batch_size=args.batch_size, 
    n_steps=args.n_steps, 
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}",
)

os.makedirs(f"models/{run.id}", exist_ok=True)

# Use separate callbacks to avoid socket issues
checkpoint_callback = CheckpointCallback(
    save_freq=25000,  # Increased frequency
    save_path=f"models/{run.id}",
    name_prefix="ppo_pendulum"
)

wandb_callback = WandbCallback(
    gradient_save_freq=0,  # Disable gradient logging
    model_save_freq=0,     # Let checkpoint_callback handle saves
    verbose=1,             # Reduced verbosity
)

callbacks = CallbackList([wandb_callback, checkpoint_callback])

# Initial training with error handling
try:
    model.learn(
        total_timesteps=args.total_timesteps, 
        callback=callbacks, 
        progress_bar=True
    )
except Exception as e:
    print(f"Warning during training: {e}")

# Continue training
for i in range(args.num_iterations):
    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=callbacks, 
            progress_bar=True, 
            reset_num_timesteps=False
        )
    except Exception as e:
        print(f"Warning during iteration {i+1}: {e}")

model.save(f"models/{run.id}/final_model")

env.close()
run.finish()

print("Training complete!")