# ARC Prize 3 (2026) — Agent57 Experiments

This repository contains my early work for **ARC Prize 3 (2026)**, the new iteration of the ARC competition introducing interactive game-based tasks.

Unlike previous ARC editions, ARC Prize 3 is designed around environments that are significantly more reinforcement-learning friendly. This project explores whether a modern distributed RL approach — specifically Agent57 — can effectively tackle these tasks.

- ARC Prize website: https://arcprize.org/

---

## Project Goal

The goal of this project is simple:

> Implement and evaluate an Agent57-style reinforcement learning system for ARC Prize 3 environments, while comparing it to ATARI environments.

This is a personal project, it will not be perfect :(

## Current Status

What is already implemented:

- Distributed off-policy training infrastructure  
- Working DQN baseline  

What is not yet implemented:

- Checkpointing
- Agent57 architecture  
- Intrinsic motivation modules  
- Meta-controller  

At the moment, the system runs as a distributed DQN setup to validate the training stack.

## Roadmap

The next steps are:

- [ ] Implement checkpoint strategy  
- [ ] Implement Agent57 components incrementally  
- [ ] Integrate intrinsic reward mechanisms  
- [ ] Add multi-policy / meta-controller logic  
- [ ] Benchmark against the DQN baseline  
- [ ] Scale distributed training  

The objective is to build the system progressively, keeping the infrastructure stable while increasing algorithmic complexity.

## Install

```bash
pip install uv
uv sync [--extra "cpu"]
```

## Usage

```bash
uv run -m rl_arc_3.main
```
