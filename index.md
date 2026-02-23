---
layout: home
title: Home
---

# Shortening the Loss Plateau
## Transformer Optimization Research Project

This project studies optimization stagnation in Transformer training, focusing on two phenomena:

- Early training-loss plateaus caused by representation collapse and slow attention structure formation
- Grokking, where models memorize data early but generalize only after long training delays

We investigate how **task diversity, initialization constraints, and optimizer choice** can accelerate learning dynamics.

GitHub: https://github.com/jetyue04/loss-plateau

---

## 🧠 Motivation
Transformers often exhibit long optimization stagnation phases during training.

We study:
- Training loss plateaus
- Grokking generalization delays
- Methods to accelerate convergence

Our hypothesis is that **diverse training objectives and constrained capacity** promote better representation learning.

---

## 📉 Loss Plateau
We replicated training-loss plateau behavior on shallow Transformers trained on algorithmic sequence tasks.

Key findings:
- Representation collapse occurs during early training
- Repetition bias appears in embeddings
- Attention structure forms slowly during plateau phases

Multi-task training significantly shortens plateau duration compared to single-task training.

### Methods

### Results

## Grokking
Grokking is a delayed generalization phenomenon where:

- Training accuracy rises quickly
- Validation accuracy remains flat for long periods

We study modular arithmetic tasks and observe that:
- Task diversity accelerates grokking
- Optimization noise can influence generalization timing

### Methods

### Results

### Summary
