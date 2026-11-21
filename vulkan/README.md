# Vulkan backend (experimental)

The first workload-matched shader is `pair_features.comp`. It mirrors
`MPSKernelLibrary.build_pair_features`:

- Input: contiguous float32 `(batch, bodies, 12)`
- Output: float32 `(batch, bodies, bodies, 28)`
- One invocation computes one batch/target/neighbor pair.
- Relative features are neighbor-minus-target x/y position and velocity.

## Compile

With the Vulkan SDK (including `glslangValidator`) installed:

```bash
./vulkan/build.sh
```

The script compiles every `.comp` file to SPIR-V under `vulkan/build/`.
The host dispatch must use `ceil(batch * bodies * bodies / 256)` workgroups
and provide the two uint push-constant values in the declared order.

## Status and performance

This is shader source only. The project does not yet include a Vulkan runtime,
Python binding, descriptor/pipeline management, or automated GPU correctness
tests. Successful SPIR-V compilation is not evidence of runtime correctness
or a speedup.

Before calling this faster than PyTorch or TensorFlow, add a Vulkan host path,
compare exact outputs against the framework reference, and benchmark synchronized
GPU execution on real Vulkan devices. Record GPU, driver, compiler, tensor shape,
warm-up count, repeat count, and median latency. Keep the MPS backend as the
supported Apple Silicon path until those checks exist.
