# FastSpeech2 Metal Length-Regulator Benchmark

Date: 2026-11-21
Hardware: Apple M1 Max
OS: macOS 26.5.2
PyTorch: 2.7.0
Benchmark: `benchmarks/benchmark_length_regulator.py`
Warm-up: 5 iterations; measured: 30 repetitions; statistic: median latency.

| Case | Batch | Source length | Hidden size | PyTorch loop forward | Metal forward | Forward speedup | PyTorch loop forward+backward | Metal forward+backward | Training-step speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small | 2 | 30 | 256 | 13.428 ms | 0.725 ms | 18.53x | 17.766 ms | 0.854 ms | 20.81x |
| Medium | 8 | 100 | 256 | 170.884 ms | 0.761 ms | 224.61x | 206.929 ms | 0.929 ms | 222.81x |
| Large | 16 | 200 | 256 | 729.697 ms | 0.801 ms | 911.50x | 881.459 ms | 1.445 ms | 610.09x |

All benchmark outputs and lengths were checked against the reference implementation before timing.

## Interpretation

These numbers measure the length-regulator operation in isolation, not full FastSpeech2 training throughput. The reference implementation performs Python loops and per-duration scalar reads, so the large speedups primarily demonstrate removal of that CPU-side bottleneck. They should not be presented as whole-model or end-to-end TTS speedups.

## Device compatibility fix retained separately

The FastSpeech2 STFT path previously forced input tensors onto CUDA with `.cuda()`. The retained patch changes the transform bases and intermediate tensors to follow the input tensor's device, allowing the same code path to operate on CPU, Apple MPS, and CUDA. CUDA execution itself was not available to validate on the M1 Mac.
