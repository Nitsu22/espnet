---
name: lab-gpu-hosts
description: このESPnet作業環境で使えるラボGPUサーバーの一覧。SSH接続先の選択、GPUの比較、実行候補の確認に使う。
---

# ラボGPUサーバー一覧

ホストの構成をまとめた一覧。現在のGPU使用率やメモリ使用量は記録せず、実行前に接続先で確認する。

| SSH接続先 | GPU |
|---|---|
| tensor | NVIDIA A100 x8 |
| ampere | NVIDIA GeForce RTX x4 |
| turing | Quadro RTX 5000 x4 |
| shannon | NVIDIA A100 80GB x8 |
| shijimi | NVIDIA RTX 6000 Ada Generation x2 |
| aurum, titanium, sushi, rhodium | 各 NVIDIA GeForce RTX 3090 x1 |
| pasmo, funnyv, tooru, suica | 各 NVIDIA GeForce RTX 4090 x1 |
| kira | NVIDIA GeForce RTX 5090 x1 |

`ampere` の詳細型番や、表にないメモリ容量が選択に必要なら、`nvidia-smi` で確認する。実行手順は `.codex/skills/lab-gpu-run/SKILL.md` を使う。
