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
## Strategy: Randomized Grid Search
To capture the complex interactions between hyperparameters, we have moved away from isolated variable testing. Each team member is assigned 5 unique combinations sampled from a multi-dimensional parameter grid.

## Student 1 (Filip): Learning Rate Variations

```bash
# Config 1.1
python train_rl.py --learning_rate 0.0003 --batch_size 128 --n_steps 2048 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --n_epochs 10 --experiment_name PPO_Rand_1.1 --wandb_key="KEY"

# Config 1.2
python train_rl.py --learning_rate 0.0001 --batch_size 64 --n_steps 4096 --gamma 0.95 --gae_lambda 0.98 --clip_range 0.1 --n_epochs 20 --experiment_name PPO_Rand_1.2 --wandb_key="KEY"

# Config 1.3
python train_rl.py --learning_rate 0.001 --batch_size 256 --n_steps 2048 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.3 --n_epochs 10 --experiment_name PPO_Rand_1.3 --wandb_key="KEY"

# Config 1.4
python train_rl.py --learning_rate 0.0005 --batch_size 128 --n_steps 4096 --gamma 0.99 --gae_lambda 0.98 --clip_range 0.2 --n_epochs 10 --experiment_name PPO_Rand_1.4 --wandb_key="KEY"

# Config 1.5
python train_rl.py --learning_rate 0.0001 --batch_size 256 --n_steps 2048 --gamma 0.95 --gae_lambda 0.95 --clip_range 0.1 --n_epochs 20 --experiment_name PPO_Rand_1.5 --wandb_key="KEY"
```

## Student 2 (Leon): Batch Size Variations

```bash
# Config 2.1
python train_rl.py --learning_rate 0.0005 --batch_size 64 --n_steps 2048 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.3 --n_epochs 20 --experiment_name PPO_Rand_2.1 --wandb_key="KEY"

# Config 2.2
python train_rl.py --learning_rate 0.0003 --batch_size 256 --n_steps 4096 --gamma 0.95 --gae_lambda 0.98 --clip_range 0.2 --n_epochs 10 --experiment_name PPO_Rand_2.2 --wandb_key="KEY"

# Config 2.3
python train_rl.py --learning_rate 0.0001 --batch_size 128 --n_steps 2048 --gamma 0.99 --gae_lambda 0.98 --clip_range 0.1 --n_epochs 10 --experiment_name PPO_Rand_2.3 --wandb_key="KEY"

# Config 2.4
python train_rl.py --learning_rate 0.001 --batch_size 64 --n_steps 4096 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --n_epochs 20 --experiment_name PPO_Rand_2.4 --wandb_key="KEY"

# Config 2.5
python train_rl.py --learning_rate 0.0003 --batch_size 128 --n_steps 2048 --gamma 0.95 --gae_lambda 0.95 --clip_range 0.3 --n_epochs 20 --experiment_name PPO_Rand_2.5 --wandb_key="KEY"
```

## Student 3 (Erik): Rollout Steps (n_steps) Variations

```bash
# Config 3.1
python train_rl.py --learning_rate 0.0001 --batch_size 256 --n_steps 4096 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --n_epochs 10 --experiment_name PPO_Rand_3.1 --wandb_key="KEY"

# Config 3.2
python train_rl.py --learning_rate 0.0005 --batch_size 128 --n_steps 2048 --gamma 0.95 --gae_lambda 0.98 --clip_range 0.1 --n_epochs 20 --experiment_name PPO_Rand_3.2 --wandb_key="KEY"

# Config 3.3
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 4096 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.3 --n_epochs 10 --experiment_name PPO_Rand_3.3 --wandb_key="KEY"

# Config 3.4
python train_rl.py --learning_rate 0.001 --batch_size 256 --n_steps 4096 --gamma 0.95 --gae_lambda 0.95 --clip_range 0.2 --n_epochs 20 --experiment_name PPO_Rand_3.4 --wandb_key="KEY"

# Config 3.5
python train_rl.py --learning_rate 0.0005 --batch_size 64 --n_steps 2048 --gamma 0.99 --gae_lambda 0.98 --clip_range 0.2 --n_epochs 10 --experiment_name PPO_Rand_3.5 --wandb_key="KEY"
```

## Student 4 (Razvan): Gamma and GAE Lambda (Discount Factors)

```bash
# Config 4.1
python train_rl.py --learning_rate 0.0001 --batch_size 128 --n_steps 4096 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.3 --n_epochs 20 --experiment_name PPO_Rand_4.1 --wandb_key="KEY"

# Config 4.2
python train_rl.py --learning_rate 0.001 --batch_size 128 --n_steps 2048 --gamma 0.95 --gae_lambda 0.98 --clip_range 0.2 --n_epochs 10 --experiment_name PPO_Rand_4.2 --wandb_key="KEY"

# Config 4.3
python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.1 --n_epochs 20 --experiment_name PPO_Rand_4.3 --wandb_key="KEY"

# Config 4.4
python train_rl.py --learning_rate 0.0005 --batch_size 256 --n_steps 4096 --gamma 0.95 --gae_lambda 0.95 --clip_range 0.3 --n_epochs 10 --experiment_name PPO_Rand_4.4 --wandb_key="KEY"

# Config 4.5
python train_rl.py --learning_rate 0.0001 --batch_size 64 --n_steps 2048 --gamma 0.99 --gae_lambda 0.98 --clip_range 0.2 --n_epochs 10 --experiment_name PPO_Rand_4.5 --wandb_key="KEY"
```

## Student 5 (Andrii): Clip Range and Epochs

```bash
# Config 5.1
python train_rl.py --learning_rate 0.0003 --batch_size 256 --n_steps 2048 --gamma 0.95 --gae_lambda 0.95 --clip_range 0.1 --n_epochs 10 --experiment_name PPO_Rand_5.1 --wandb_key="KEY"

# Config 5.2
python train_rl.py --learning_rate 0.0005 --batch_size 128 --n_steps 4096 --gamma 0.99 --gae_lambda 0.98 --clip_range 0.3 --n_epochs 20 --experiment_name PPO_Rand_5.2 --wandb_key="KEY"

# Config 5.3
python train_rl.py --learning_rate 0.001 --batch_size 64 --n_steps 2048 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --n_epochs 10 --experiment_name PPO_Rand_5.3 --wandb_key="KEY"

# Config 5.4
python train_rl.py --learning_rate 0.0001 --batch_size 256 --n_steps 4096 --gamma 0.95 --gae_lambda 0.98 --clip_range 0.2 --n_epochs 20 --experiment_name PPO_Rand_5.4 --wandb_key="KEY"

# Config 5.5
python train_rl.py --learning_rate 0.0005 --batch_size 64 --n_steps 2048 --gamma 0.95 --gae_lambda 0.95 --clip_range 0.1 --n_epochs 10 --experiment_name PPO_Rand_5.5 --wandb_key="KEY"
```

## Parameters Explained (Grid Boundaries)
- **Learning Rate (1e-4 to 1e-3):** Balances convergence speed vs. stability.
- **Batch Size (64 to 256):** Affects gradient estimate quality and training throughput.
- **N Steps (2048 to 4096):** Determines how much experience is gathered before an update.
- **Gamma & Lambda:** Control the trade-off between bias and variance in reward estimation.
- **Clip Range (0.1 to 0.3):** Limits how much the policy can change in one update (PPO stability).

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
