# Training throughput across machines

Sustained tokens/s (median step over the second half of each run). Same model, data, order and hyperparameters everywhere; per-machine details in `runs/<machine>/REPORT.md`.

| Run | macbook-air-m5-32gb | rtx-5090-laptop-gpu-24gb |
|---|---|---|
| mlx-fp32 | 528 (mb 1) | – |
| torch-fp32 | 399 (mb 1) | 3,664 (mb 1) |
| mlx-bf16 | 1,655 (mb 1) | – |
| torch-bf16 | 673 (mb 1) | 10,033 (mb 1) |
| mlx-fp32-tf32 | 1,049 (mb 1) | – |
| torch-fp32-tf32 | – | 5,112 (mb 1) |

Final training loss (should agree across machines for the same precision):

| Run | macbook-air-m5-32gb | rtx-5090-laptop-gpu-24gb |
|---|---|---|
| mlx-fp32 | 1.7363 | – |
| torch-fp32 | 1.7364 | 1.7365 |
| mlx-bf16 | 1.7535 | – |
| torch-bf16 | 1.7532 | 1.7531 |
| mlx-fp32-tf32 | 1.7362 | – |
| torch-fp32-tf32 | – | 1.7362 |
