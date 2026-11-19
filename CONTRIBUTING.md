# Contributing

Thanks for helping improve Compute Shaders.

## Before opening a pull request

1. Keep kernels small and document their input/output layout.
2. Add a correctness test against a clear PyTorch reference.
3. Test edge cases such as the smallest valid tensor and non-contiguous inputs.
4. Benchmark with MPS synchronization around timed operations.
5. Report the exact Mac model, macOS version, and PyTorch version with performance results.

## Performance claims

A custom kernel is not automatically faster. Include:
- the reference implementation,
- tensor shapes and dtypes,
- warm-up and repeat counts,
- median timings,
- correctness tolerance,
- device and software versions.

Do not report only the fastest run.

## Numerical behavior

Prefer exact behavior where practical. If a kernel changes reduction order or uses approximate math, document the expected tolerance and test it.

## Scope

The current supported backend is PyTorch MPS on Apple Silicon. Vulkan files are experimental until a runtime and integration tests are available.
