from stable_baselines3 import PPO
from ot2_env import OT2ENV
import numpy as np
import matplotlib.pyplot as plt

# Initialise the simulation environment
num_agents = 1
env = OT2ENV(num_agents, render_mode = "human")
obs, info = env.reset()

### 
# 
# Do all the CV things so that you end up with a list of goal positions
#
###

goal_positions = [[-0.1275, -0.1101, 0.1903], [-0.0775, -0.0101, 0.1303]]

# Store all trajectories
all_trajectories = []

# Load the trained agent
model = PPO.load("robotics\\final_model")
   
# Get robot key
robot_key = list(info["pipette_positions"].keys())[0]

for goal_pos in goal_positions:
    # Lists to store data for this trajectory
    x_history = []
    y_history = []
    z_history = []
    
    # Target coordinates
    env.target = np.array(goal_pos)
    

    # Run the control algorithm until the robot reaches the goal position
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        
        # Get current pipette position and store it
        pos = info["pipette_positions"][robot_key]
        x_history.append(pos[0])
        y_history.append(pos[1])
        z_history.append(pos[2])
        
        error = np.linalg.norm(obs[6:9])
        
        # Drop the inoculum if the robot is within the required error
        if error < 0.005:
            print(f"Target {goal_pos} reached!")
            break
    
    # Store this trajectory
    all_trajectories.append({
        'x': x_history,
        'y': y_history,
        'z': z_history,
        'target': goal_pos
    })

# Plot all trajectories at the end
for i, traj in enumerate(all_trajectories):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(traj['x'])
    plt.axhline(y=traj['target'][0], color='r', linestyle='--', label='target')
    plt.ylabel('X')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(traj['y'])
    plt.axhline(y=traj['target'][1], color='r', linestyle='--', label='target')
    plt.ylabel('Y')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(traj['z'])
    plt.axhline(y=traj['target'][2], color='r', linestyle='--', label='target')
    plt.ylabel('Z')
    plt.xlabel('Time step')
    plt.legend()
    
    plt.suptitle(f'Trajectory {i+1}: target {traj["target"]}')
    plt.tight_layout()

plt.show()
