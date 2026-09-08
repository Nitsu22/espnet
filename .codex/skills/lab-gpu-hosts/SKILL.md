---
name: lab-gpu-hosts
description: Inventory of lab GPU SSH targets for this ESPnet workspace. Use when Codex needs to choose or name lab GPU hosts, compare GPU classes, or list candidate SSH targets before running lab GPU jobs.
---

# Lab GPU Hosts

Use this as stable host inventory only. Do not record current GPU utilization or current memory usage here; check live usage before running.

| SSH target | GPU |
|---|---|
| tensor | NVIDIA A100 x8 |
| ampere | NVIDIA GeForce RTX x4 |
| turing | Quadro RTX 5000 x4 |
| shannon | NVIDIA A100 80GB x8 |
| shijimi | NVIDIA RTX 6000 Ada Generation x2 |
| aurum, titanium, sushi, rhodium | NVIDIA GeForce RTX 3090 x1 each |
| pasmo, funnyv, tooru, suica | NVIDIA GeForce RTX 4090 x1 each |
| kira | NVIDIA GeForce RTX 5090 x1 |
