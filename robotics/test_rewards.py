from stable_baselines3 import PPO
from ot2_env import OT2ENV

# Training
env = OT2ENV(render_mode='human')
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
env.close()

# Save the model
# model.save("ppo_ot2_test")

# Testing
env = OT2ENV(render_mode='human') 
obs, info = env.reset()

for i in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    
    print(f"Step {i}: Reward={reward:.3f}, Distance={info['distance_to_target']:.6f}m")
    
    if done:
        print(f"Episode finished! Final distance: {info['distance_to_target']:.6f}m")
        obs, info = env.reset()

env.close()