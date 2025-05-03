# Particle Swarm Optimization (PSO) Based CartPole Control Solution

## Project Overview
This repository provides a complete implementation of solving the OpenAI Gym CartPole-v1 control problem using Particle Swarm Optimization (PSO) algorithm. Key features include:：
-  **Gradient-free optimization**: Searches for optimal policy parameters through swarm intelligence
-  **Parameter saving**: Automatically saves the best policy parameters obtained from training
-  **Real-time rendering**: Visualizes the control performance after training

## Quick Start

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```
```bash
# Run training script
python pso_cartpole.py
```
```bash
# Run rendering script
python render_cartpole.py
```

## Key Parameters (pso_cartpole)
- **Number of particles**: `num_particles`，default 30
- **Generations**: `generations`，default 50
- **Inertia weight**: `w`，default 0.729
