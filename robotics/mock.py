import gymnasium as gym
import numpy as np
import ot2_env
from pid import PIDController

targets = [
    [0.1145, 0.0670, 0.1215],
    [0.1491, 0.1051, 0.1215],
    [0.1837, 0.1432, 0.1215],
    [0.2183, 0.1814, 0.1215],
    [0.2440, 0.1900, 0.1215],
]

pid_x = PIDController(kp=9.0, ki=0.048, kd=0.0)
pid_y = PIDController(kp=9.0, ki=0.048, kd=0.0)
pid_z = PIDController(kp=5.0, ki=0.024, kd=0.0)

dt = 1.0 / 240.0
max_steps = 5000
drop_threshold = 0.001

env = gym.make("OT2ENV-v0", max_steps=max_steps, render_mode="human")
obs, info = env.reset(seed=42)

target_idx = 0
dwell_counter = 0
dwell_required = 10

print(f"Moving to target {target_idx}")

for step in range(max_steps):
    if target_idx >= len(targets):
        print(f"All targets completed at step {step}!")
        break

    target = np.array(targets[target_idx])
    
    # Get actual pipette position from info dict (raw coordinates)
    pipette_pos = np.array(info["pipette_positions"]["robotId_2"])
    
    # Calculate error in raw coordinates
    error = target - pipette_pos
    error_x, error_y, error_z = error
    distance = np.linalg.norm(error)
    
    # Compute PID control
    control_x = pid_x.compute(error_x, dt, output_limits=(-1.0, 1.0))
    control_y = pid_y.compute(error_y, dt, output_limits=(-1.0, 1.0))
    control_z = pid_z.compute(error_z, dt, output_limits=(-1.0, 1.0))
    
    if distance <= drop_threshold:
        # At target - hold position
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        dwell_counter += 1
        
        if dwell_counter >= dwell_required:
            print(f"Completed target {target_idx} at step {step}, distance: {distance:.6f}")
            target_idx += 1
            dwell_counter = 0
            pid_x.reset()
            pid_y.reset()
            pid_z.reset()
            if target_idx < len(targets):
                print(f"Moving to target {target_idx}")
    else:
        # Still approaching
        action = np.array([control_x, control_y, control_z], dtype=np.float32)
        dwell_counter = 0
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

env.close()
print(f"Finished! Completed {target_idx} targets")