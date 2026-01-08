from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TrainingStatsCallback(BaseCallback):
    """Callback to track and print training progress."""
    
    def __init__(self, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.final_distances = []
        self.min_distances = []
        self.successes = 0
        self.out_of_bounds = 0
        self.timeouts = 0
        self.total_episodes = 0
        
        # Track current episode
        self.current_reward = 0
        self.current_length = 0
        self.current_min_distance = float('inf')
    
    def _on_step(self) -> bool:
        # Track episode stats
        self.current_reward += self.locals["rewards"][0]
        self.current_length += 1
        
        # Get current distance from info
        info = self.locals["infos"][0]
        if "distance_to_target" in info:
            dist = info["distance_to_target"]
            self.current_min_distance = min(self.current_min_distance, dist)
        
        # Check if episode ended
        done = self.locals["dones"][0]
        if done:
            self.total_episodes += 1
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.min_distances.append(self.current_min_distance)
            
            # Determine outcome
            final_dist = info.get("distance_to_target", float('inf'))
            self.final_distances.append(final_dist)
            
            env = self.training_env.envs[0]
            if final_dist <= env.target_threshold:
                self.successes += 1
            elif self.current_length >= env.max_steps:
                self.timeouts += 1
            else:
                self.out_of_bounds += 1
            
            # Reset tracking
            self.current_reward = 0
            self.current_length = 0
            self.current_min_distance = float('inf')
        
        # Print stats periodically
        if self.n_calls % self.check_freq == 0:
            self._print_stats()
        
        return True
    
    def _print_stats(self):
        if self.total_episodes == 0:
            return
        
        recent = min(100, self.total_episodes)
        avg_reward = np.mean(self.episode_rewards[-recent:])
        avg_length = np.mean(self.episode_lengths[-recent:])
        avg_min_dist = np.mean(self.min_distances[-recent:])
        avg_final_dist = np.mean(self.final_distances[-recent:])
        
        print(f"\n{'='*60}")
        print(f"Timestep: {self.n_calls} | Episodes: {self.total_episodes}")
        print(f"Last {recent} episodes:")
        print(f"  Avg Reward:     {avg_reward:.2f}")
        print(f"  Avg Length:     {avg_length:.1f}")
        print(f"  Avg Min Dist:   {avg_min_dist:.6f}m")
        print(f"  Avg Final Dist: {avg_final_dist:.6f}m")
        print(f"Outcomes (all episodes):")
        print(f"  Successes:     {self.successes} ({100*self.successes/self.total_episodes:.1f}%)")
        print(f"  Out of Bounds: {self.out_of_bounds} ({100*self.out_of_bounds/self.total_episodes:.1f}%)")
        print(f"  Timeouts:      {self.timeouts} ({100*self.timeouts/self.total_episodes:.1f}%)")
        print(f"{'='*60}\n")
