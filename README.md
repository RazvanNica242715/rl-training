# OT-2 Reinforcement Learning Control: Training & Reward Engineering

This project documents the collaborative effort of our group of 5 to train a Reinforcement Learning (RL) agent for precise control of the OT-2 liquid handling robot within the NPEC Automated Plant Root Inoculation Pipeline. Using the Gymnasium-compatible wrapper for simulation environment, developed by Razvan Nica, we iterated on reward function designs to achieve sub-millimeter precision and stable positioning.

---

## 1. Evolution of the Reward Function

The training process focused on overcoming two primary behavioral hurdles through iterative reward engineering:

### Phase 1: Overcoming the "Precision Gap"
Initially, the agent utilized a simple reward structure based on `-distance` plus basic thresholds. 
* **The Issue**: While the agents learned to get close to the target, they consistently stalled approximately 2-5mm away from the goal.
* **The Cause**: The reward gradient wasn't strong enough at close range to incentivize the final, precise adjustments required to reach the exact coordinate.
* **The Solution**: We shifted to a **dense reward function** that incorporates an exponential proximity bonus: $e^{-d/0.01} \times 0.2$.

### Phase 2: Combating "Milestone Farming"
To encourage the agent to reach specific distances, we introduced milestone bonuses.
* **The Issue**: Agents learned to exploit the system by moving back and forth across a milestone boundary to "farm" the reward repeatedly.
* **The Solution**: 
    1. **One-Time Milestones**: We implemented a tracking system (`self._milestones_reached`) to ensure each bonus (10mm, 5mm, 2mm, 1mm) is only granted once per episode.
    2. **Dwell Requirement**: We updated the termination condition so that the robot must stay within the target threshold for **10 consecutive steps** (`dwell_steps`) rather than just touching it.

---

## 2. Final Reward Structure

The resulting reward function encourages efficient movement, high precision, and final stability:

| Component | Description | Logic / Weight |
| :--- | :--- | :--- |
| **Distance Delta** | Reward for moving closer to the target between steps. | $\text{clip}(\Delta d \times 100, -1, 1)$ |
| **Proximity Well** | Exponential reward that spikes as the agent enters the final centimeters. | $\exp(-dist / 0.01) \times 0.2$ |
| **Milestone Bonuses** | One-time rewards for reaching 10mm, 5mm, 2mm, and 1mm thresholds. | $+1.0$ to $+4.0$ (Once per episode) |
| **Dwell Reward** | Rewards the agent for remaining stationary and stable at the goal. | $0.3 + (0.7 \times \text{progress})$ |
| **Success Bonus** | A large terminal reward for completing the 10-step dwell requirement. | $+20.0$ |
| **Penalties** | Costs for time, leaving the workspace, or high velocity near the target. | Boundary: $-3.0 - 50 \times \text{overshoot}$ |

---

## 3. Training Methodology

The team performed a distributed hyperparameter optimization to find the most robust control policy:

* **Collaborative Grid Search**: We created a grid of values and each of the 5 students randomly sampled 5 combinations, resulting in **25 total training runs**.
* **Remote Monitoring**: We used **WandB (Weights & Biases)** and **ClearML** to track experiments and monitor agent performance in real-time.
* **Model Selection**: After training, all models were evaluated via an automated testing script to determine the highest performing agent.

---

## 4. Best Performing Model

The evaluation process identified **Run 4.4** as the superior model, demonstrating the best balance of speed, accuracy, and lack of oscillation.

* **Model Path**: `/robotics/artifacts/models/model.zip`
* **Key Stats**:
    * **Accuracy**: < 1mm positioning
    * **Stability**: Zero oscillation during the dwell phase

---