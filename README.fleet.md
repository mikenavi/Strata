# mikenavi Strata native component

This branch is the native engine component based on upstream1aff6d14, engine
0.1.50. Bounded checkpoint chains/LCP divergence learning and serial/slot token
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
