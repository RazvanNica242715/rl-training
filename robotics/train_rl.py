"""
RL Training Script for OT-2 Controller

Uses Weights & Biases for experiment tracking and ClearML for remote training.
"""

import gymnasium as gym
import numpy as np
import wandb
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from ot2_env import OT2ENV

# For ClearML remote training
from clearml import Task

def parse_args():
    """Parse command line arguments for hyperparameters"""
    parser = argparse.ArgumentParser()
    
    # wandb key
    parser.add_argument("--wandb_key", type=str, default="")
    
    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    
    # Additional PPO parameters
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    
    # Network architecture
    parser.add_argument("--net_arch", type=str, default="64,64")
    
    # Training settings
    parser.add_argument("--total_timesteps", type=int, default=5000000)
    parser.add_argument("--save_freq", type=int, default=50000)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    
    # Experiment naming
    parser.add_argument("--experiment_name", type=str, default="PPO_Experiment")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # WANDB KEY - for this specific case, NOT RECOMMENDED
    os.environ['WANDB_API_KEY'] = args.wandb_key

    # Initialize ClearML (for remote training)
    task = Task.init(
        project_name='Mentor Group - Dean/Group 1',
        task_name=args.experiment_name
    )

    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages(['tensorboard', 'clearml', 'wandb'])
    task.execute_remotely(queue_name="default")

    # Initialize Weights & Biases
    run = wandb.init(
        project="ot2-rl-control",
        name=args.experiment_name,
        config=vars(args),
        sync_tensorboard=True,
        monitor_gym=True
    )
    
    # Create environment
    env = OT2ENV(
        num_agents=1,
        render_mode=None,
        max_steps=args.max_episode_steps
    )
    
    # Print configuration
    print("Training Configuration:")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Parse network architecture
    net_arch = [int(x) for x in args.net_arch.split(',')]
    policy_kwargs = dict(net_arch=net_arch)
    
    print(f"\nNetwork architecture: {net_arch}")

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs/{run.id}"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=f"models/{run.id}",
        name_prefix="ot2_ppo"
    )
    
    wandb_callback = WandbCallback(
        model_save_freq=args.save_freq,
        model_save_path=f"models/{run.id}",
        verbose=2
    )

    reward_callback = RewardLoggingCallback(verbose=1)

    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, wandb_callback, reward_callback],
        progress_bar=True,
        tb_log_name=f"runs/{run.id}"
    )
    
    # Save final model
    final_path = f"models/{run.id}/final_model"
    model.save(final_path)
    print(f"\nModel saved: {final_path}")
    
    # Upload to ClearML
    task.upload_artifact("final_model", artifact_object=f"{final_path}.zip")
    
    print("\nTraining complete!")
    
    env.close()
    run.finish()


class RewardLoggingCallback(BaseCallback):
    """
    Custom callback for logging environment metrics to wandb
    """
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.step_count = 0
        self.episode_rewards = []
        self.episode_steps = []
        
    def _on_step(self) -> bool:
        """
        Called after each environment step
        """
        self.step_count += 1
        
        # Access the environment (unwrap if needed)
        env = self.training_env.envs[0]
        
        # Log step-level metrics
        if hasattr(env, 'current_reward'):

            wandb.log({
                "step/current_reward": env.current_reward,
                "step/current_step": env.current_step,
                "step/global_step": self.step_count,
            })
                
        return True

if __name__ == "__main__":
    main()