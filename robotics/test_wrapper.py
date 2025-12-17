"""
Test script for OT2ENV Gymnasium wrapper.

This script tests the environment by running random actions for multiple episodes
and validates observation/action spaces, reward behavior, and termination conditions.

Usage:
    python test_wrapper.py
"""

from ot2_env import OT2ENV
import numpy as np
# import matplotlib.pyplot as plt

def main():
    # Initialize environment
    env = OT2ENV(num_agents=1, render_mode='human', max_steps=1000)

    print("Testing OT2ENV Environment")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print()

    # Tracking variables
    num_episodes = 5
    all_rewards = []
    episode_lengths = []
    distances = []
    success_count = []

    # Run test episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        steps = 0
        episode_reward = 0

        print(f"  Episode: {episode + 1}, Target: {env.target}")
        
        while not done:
            # Random action
            action = env.action_space.sample()
            print(f"    Step: {step + 1}, Action: {action}")

            # Perform a step
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"    Reward: {reward}")

            # Track metrics
            all_rewards.append(reward)
            episode_reward += reward
            distances.append(info['distance_to_target'])
            steps +=1
            done = terminated or truncated

            # Validate spaces
            assert env.observation_space.contains(obs), "Observation out of"
            assert env.action_space.contains(action), "Action out of bounds!"

        episode_lengths.append(steps)
        if terminated:
            success_count += 1
            print(f"  Success in {steps} steps (distance: {info['distance_to_target']:.6f}m)")
        else:
            print(f"  Truncated at {steps} steps (distance: {info['distance_to_target']:.6f}m)")

    # Summary
    print("SUMMARY")
    print(f"  Episodes: {num_episodes}")
    print(f"  Successful: {success_count}")
    print(f"  Avg episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Reward - Mean: {np.mean(all_rewards):.3f}, Std: {np.std(all_rewards):.3f}")
    print(f"  Min distance achieved: {np.mean(distances):.6f}m")

    #  # Create plots
    # fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # axes[0].plot(all_rewards)
    # axes[0].set_title('Rewards Over Time')
    # axes[0].set_xlabel('Step')
    # axes[0].set_ylabel('Reward')
    # axes[0].grid(True, alpha=0.3)
    
    # axes[1].bar(range(num_episodes), episode_lengths)
    # axes[1].set_title('Episode Lengths')
    # axes[1].set_xlabel('Episode')
    # axes[1].set_ylabel('Steps')
    # axes[1].grid(True, alpha=0.3)
    
    # axes[2].plot(distances)
    # axes[2].axhline(y=0.01, color='r', linestyle='--', label='Threshold')
    # axes[2].set_title('Distance to Target')
    # axes[2].set_xlabel('Step')
    # axes[2].set_ylabel('Distance (m)')
    # axes[2].legend()
    # axes[2].grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()

    env.close()
    print("Test complete!")

if __name__ == '__main__':
    main()