# Test Matrix-Vector / Vector-Vector operations in MKL and Rust

## Requirements:
 - Rust Nightly Build (`rustup update && rustup install nightly`)
 - [Intel MKL](https://software.intel.com/en-us/mkl)
   - Modify `intel-mkl-src` and `intel-mkl-tool` to use `mkl_intel_thread.lib` and `libiomp5md.lib` for the best performance.

## Performance Results (in milliseconds):
 - Run command: `RUSTFLAGS="-C target-feature=+avx2" cargo +nightly run --release`
 - My machine is AMD Ryzen Threadripper 1950X 16-Core @ 4.00GHz with 128 GB Ram.


### vector-vector operations

|                        300 millions double. (about 2.235GB)                |||||
|:------------------------:|:-----------:|:-----------:|:-----------:|:---------:|
|                                   **Dot Product**                         |||||
| Programs                 | First Time  | Second Time | Third Time  | Avg. Time |
| Intel MKL cblas_ddot     |   109.596   |   101.785   |   91.277    |  100.886  |
| Rust in Rayon            |    87.727   |    83.21    |   85.703    |   85.546  |
| Rust in Rayon and SIMD   |    74.327   |    73.081   |   73.905    |   73.771  |
||||||
|                    **Element-wise Addition of Two Vectors**               |||||
| Intel MKL cblas_ddot     |  468.4062   |   438.7717  |   440.6319  |  449.2699 |
| Rust (SIMD with f64x4)   |  434.8226   |   428.7067  |   425.4852  |  429.6715 |
| Rust ( SIMD with f64x8)  |  430.687    |   430.8837  |   425.9078  |  429.1595 |


|                        2.1 billions double. (about 15.646GB)               |||||
|:------------------------:|:-----------:|:-----------:|:-----------:|:---------:|
|                                   **Dot Product**                         |||||
| Programs                 | First Time  | Second Time | Third Time  | Avg. Time |
| Intel MKL cblas_ddot     |   636.793   |   579.091   |  648.999    |  621.627  |
| Rust in Rayon            |   523.24    |   520.572   |  517.373    |  520.395  |
| Rust in Rayon and SIMD   |   514.488   |   508.294   |  506.511    |  509.764  |
||||||
|                    **Element-wise Addition of Two Vectors**               |||||
| Intel MKL cblas_ddot     |  3114.2552  |  3080.1915  |  3082.665   | 3092.3706 |
| Rust (SIMD with f64x4)   |  3018.102   |  3023.3982  |  3017.301   | 3019.6004 |
| Rust ( SIMD with f64x8)  |  3000.0463  |  2994.6608  |  3009.3949  | 3001.3673 |

### matrix-vector operations

|                          (2000 x 50000) vs (50000 x 1)                     |||||
|:------------------------:|:-----------:|:-----------:|:-----------:|:---------:|
|                                **Multiplication**                         |||||
| Programs                 | First Time  | Second Time | Third Time  | Avg. Time |
| Intel MKL cblas_dgemv    |   39.133    |    39.255   |   39.157    |   39.182  |
| Rust in Rayon            |   12.587    |    12.907   |   12.651    |   12.713  |
