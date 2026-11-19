# Vulkan backend (experimental)

This directory preserves the original Vulkan shader experiments.

The current project does not yet include a Vulkan runtime, Python binding, or automated GPU correctness tests. The sample shader and build script only demonstrate GLSL-to-SPIR-V compilation when the Vulkan SDK is installed.

Do not treat this backend as production-ready. The supported implementation is currently Apple Metal through PyTorch MPS.
