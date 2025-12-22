# RL Training - Dean Group 1


## Setting Up

Weights & Biases is a cloud-based platform, so you don’t need to install anything on your local machine (you can run a W&B server locally as well if you prefer). However, you do need to install the W&B Python package in your Python environment. Here’s how to get started:

**Installation:**

```bash
pip install wandb
```

**Create an Account:**
- Sign up at the [W&B website](https://wandb.ai/) to create your account.

**Authentication:**
- After signing up, use the following command in your terminal:

```bash
wandb login
```

- A browser window will open asking you to authenticate your account - or you will be asked to provide an API which you can create on the W&B website in your user account settings.

Install the ClearML package and initialize the configuration file. Run the following commands in the terminal (make sure you are in the correct virtual environment):

```bash
pip install clearml

clearml-init
```

Copy these credentials when prompted:

```
api { 
    web_server: http://194.171.191.227:8080
    api_server: http://194.171.191.227:8008
    files_server: http://194.171.191.227:8081
    # Students
    credentials {
        "access_key" = "UJODYOFVELU1XCB7OFM2FKU7XCY48K"
        "secret_key"  = "OKCS8xT-vngmYWgpIMYsu_GbS2fLgMmMp1MbzqyLZdWZtA-FGxlUJ5KGISFGPMdcDDk"
    }
}
```

Leave all other settings as default.

Note: You need to be connected to the VPN to queue jobs and access the dashboard.

---

## Student 1 (Filip): Learning Rate Variations

```bash
# Config 1.1: Baseline
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --n_epochs 10 --experiment_name PPO_Experiment_1.1

# Config 1.2: Lower LR
python train_rl.py --learning_rate 0.0001 --batch_size 64 --n_steps 2048 --n_epochs 10 --experiment_name PPO_Experiment_1.2

# Config 1.3: Higher LR
python train_rl.py --learning_rate 0.001 --batch_size 64 --n_steps 2048 --n_epochs 10 --experiment_name PPO_Experiment_1.3

# Config 1.4: Very high LR
python train_rl.py --learning_rate 0.003 --batch_size 64 --n_steps 2048 --n_epochs 10 --experiment_name PPO_Experiment_1.4

# Config 1.5: Adaptive (lower LR + more epochs)
python train_rl.py --learning_rate 0.0001 --batch_size 64 --n_steps 2048 --n_epochs 20 --experiment_name PPO_Experiment_1.5
```

**Focus:** Find optimal learning rate (0.0001 to 0.003 range)

## Student 2 (Leon): Batch Size Variations

```bash
# Config 2.1: Small batch
python train_rl.py --learning_rate 0.0003 --batch_size 32 --n_steps 2048 --n_epochs 10 --experiment_name PPO_Experiment_2.1

# Config 2.2: Baseline
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --n_epochs 10 --experiment_name PPO_Experiment_2.2

# Config 2.3: Large batch
python train_rl.py --learning_rate 0.0003 --batch_size 128 --n_steps 2048 --n_epochs 10 --experiment_name PPO_Experiment_2.3

# Config 2.4: Very large batch + adjusted LR
python train_rl.py --learning_rate 0.0005 --batch_size 256 --n_steps 2048 --n_epochs 10 --experiment_name PPO_Experiment_2.4

# Config 2.5: Large batch + more epochs
python train_rl.py --learning_rate 0.0003 --batch_size 128 --n_steps 2048 --n_epochs 15 --experiment_name PPO_Experiment_2.5
```

**Focus:** Find optimal batch size (32 to 256 range) and interaction with epochs

## Student 3 (Erik): Rollout Steps (n_steps) Variations

```bash
# Config 3.1: Short rollouts
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 1024 --n_epochs 10 --experiment_name PPO_Experiment_3.1

# Config 3.2: Baseline
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --n_epochs 10 --experiment_name PPO_Experiment_3.2

# Config 3.3: Long rollouts
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 4096 --n_epochs 10 --experiment_name PPO_Experiment_3.3

# Config 3.4: Very long rollouts
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 8192 --n_epochs 10 --experiment_name PPO_Experiment_3.4

# Config 3.5: Long rollouts + fewer epochs
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 4096 --n_epochs 5 --experiment_name PPO_Experiment_3.5
```

**Focus:** Find optimal experience collection (1024 to 8192 steps per update)

## Student 4 (Razvan): Gamma and GAE Lambda (Discount Factors)

```bash
# Config 4.1: Low gamma (prioritize immediate rewards)
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --gamma 0.90 --gae_lambda 0.90 --experiment_name PPO_Experiment_4.1

# Config 4.2: Medium-low gamma
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --gamma 0.95 --gae_lambda 0.95 --experiment_name PPO_Experiment_4.2

# Config 4.3: Baseline (high gamma)
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --gamma 0.99 --gae_lambda 0.95 --experiment_name PPO_Experiment_4.3

# Config 4.4: Very high gamma
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --gamma 0.995 --gae_lambda 0.98 --experiment_name PPO_Experiment_4.4

# Config 4.5: Mismatched (high gamma, low lambda)
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --gamma 0.99 --gae_lambda 0.85 --experiment_name PPO_Experiment_4.5
```

**Focus:** Find optimal discount factors (how much to value future rewards)

## Student 5 (Andrii): Clip Range and Epochs

```bash
# Config 5.1: Conservative clipping
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --clip_range 0.1 --n_epochs 10 --experiment_name PPO_Experiment_5.1

# Config 5.2: Baseline
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --clip_range 0.2 --n_epochs 10 --experiment_name PPO_Experiment_5.2

# Config 5.3: Aggressive clipping
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --clip_range 0.3 --n_epochs 10 --experiment_name PPO_Experiment_5.3

# Config 5.4: Many epochs (more optimization)
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --clip_range 0.2 --n_epochs 20 --experiment_name PPO_Experiment_5.4

# Config 5.5: Few epochs (faster updates)
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --clip_range 0.2 --n_epochs 5 --experiment_name PPO_Experiment_5.5
```

**Focus:** Find optimal PPO-specific parameters (clipping and optimization iterations)

##  Network Architecture (If Time - Assign Name and Notify Others)

```bash
# Smaller network (faster)
python train_rl.py --learning_rate 0.0003 --net_arch "32,32"

# Baseline
python train_rl.py --learning_rate 0.0003 --net_arch "64,64"

# Larger network (more capacity)
python train_rl.py --learning_rate 0.0003 --net_arch "128,128"

# Deeper network
python train_rl.py --learning_rate 0.0003 --net_arch "64,64,64"

# Wide and deep
python train_rl.py --learning_rate 0.0003 --net_arch "128,128,64"
```

## Summary Table

| Student | Focus Area | Key Parameters Varied |
|---------|-----------|---------------------|
| 1 | Learning Rate | 0.0001, 0.0003, 0.001, 0.003 |
| 2 | Batch Size | 32, 64, 128, 256 |
| 3 | Rollout Steps | 1024, 2048, 4096, 8192 |
| 4 | Discount Factors | gamma: 0.90-0.995, gae_lambda: 0.85-0.98 |
| 5 | Clipping & Epochs | clip: 0.1-0.3, epochs: 5-20 |

## Parameters

**Learning Rate:**
- Too low -> Learns very slowly, may not converge in 1M steps
- Too high -> Unstable, policy oscillates, never settles

**Batch Size:**
- Small (32) -> Noisy gradients, faster updates, more exploration
- Large (256) -> Stable gradients, slower updates, less exploration

**N Steps (Rollout Length):**
- Short (1024) -> Frequent updates, less data per update
- Long (8192) -> Infrequent updates, more data per update

**Gamma:**
- Low (0.90) -> Focus on immediate rewards (get to target fast, ignore path)
- High (0.995) -> Consider long-term (smooth trajectory matters)

**Clip Range:**
- Low (0.1) -> Conservative policy updates, stable but slow
- High (0.3) -> Aggressive updates, faster but risky

**N Epochs:**
- Few (5) -> Quick updates, may underfit
- Many (20) -> Thorough optimization, may overfit