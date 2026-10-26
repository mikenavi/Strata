# mikenavi Strata native component

This branch is the native engine component based on upstream4b9e05a2. Its CMake
project version is0.1.51; the upstream Qwen binary package remains0.1.50. Bounded checkpoint chains/LCP divergence learning and serial/slot token
probabilities are committed features. The superseded adaptive-tail candidate is
excluded. This branch is pinned by the application's local/components.lock.json.

`bash local/build-native.sh --verify-only` verifies source bytes. To build,
provide SDK_ROOT, GGML_SOURCE and BUILD_DIR in an isolated SDK environment.
The application branch mikenavi/stable provides the release assembler. Native
custom builds are distinct from signed official components; no upstream
signature is reused. Engine and application versions are independent.

Native cache is opt-in. Preserve memory-floor/unknown-telemetry/allocation-failure
behavior and slot/main/parked ownership. Prior HIP checks do not qualify a new
compiler, CUDA/Metal/SYCL backend or full task workload.

The upstream main merge preserves all custom cache/probability code. Relative to
the previous fleet engine snapshot, generate.cpp changes one comment only; the
CMake project version changes from0.1.50 to0.1.51. DeepSeek CUDA, Metal and SYCL
changes were not backend-qualified here. A fresh linked HIP build and GPU checks
remain separate from the source sync and CPU regressions.
