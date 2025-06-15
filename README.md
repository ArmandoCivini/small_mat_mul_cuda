# small_mat_mul_cuda

This repository contains a collection of CUDA scripts implementing state-of-the-art algorithms for small matrix multiplication, based on the latest research and best-ranked schemas. The schemas themselves are not original work; they are derived from the following papers and repositories:

- [Fast Matrix Multiplication Schemes from SAT Solvers (arXiv:2505.05896)](https://arxiv.org/pdf/2505.05896)
- [A Database of Fast Matrix Multiplication Schemes (arXiv:2502.04514)](https://arxiv.org/abs/2502.04514)
- [jakobmoosbauer/symmetric-flips](https://github.com/jakobmoosbauer/symmetric-flips)
- [mkauers/matrix-multiplication](https://github.com/mkauers/matrix-multiplication)

All code in this repository is released under the GNU General Public License (GPL). If you find this code useful or use it in your own work, I would be delighted if you leave a message or let me know!

## Benchmark Results (NVIDIA GeForce RTX 4050)

| Kernel                   | Time (ms) per 10M x 5x5 mults | Speedup vs Naive |
|--------------------------|-------------------------------|------------------|
| matmul5x5_opt            | 29.92                         | 1.54x            |
| matmul5x5_opt_reg_bound  | 27.18                         | 1.70x            |
| naive                    | 46.12                         | 1.00x            |

Tested on: NVIDIA GeForce RTX 4050, CUDA 12.8, 10,000,000 independent 5x5 multiplications, averaged over 10 runs.