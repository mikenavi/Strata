#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
python3 "$repo/local/verify-source.py"
[[ "${1:-}" == --verify-only ]] && exit 0
: "${SDK_ROOT:?Set the pinned ROCm SDK root}"
: "${GGML_SOURCE:?Set the pinned external ggml source directory}"
: "${BUILD_DIR:?Set a separate build directory}"
export ROCM_PATH="$SDK_ROOT" HIP_PATH="$SDK_ROOT" HIP_PLATFORM=amd
export PATH="$SDK_ROOT/bin:$SDK_ROOT/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$SDK_ROOT/lib:$SDK_ROOT/lib/rocm_sysdeps/lib:$SDK_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
cmake -S "$repo" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_GGML_DIR="$GGML_SOURCE" -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DCMAKE_HIP_ARCHITECTURES="${GPU_ARCH:-gfx1151}" -DCMAKE_HIP_COMPILER="$SDK_ROOT/lib/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$SDK_ROOT" "-DCMAKE_PREFIX_PATH=$SDK_ROOT;$SDK_ROOT/lib/rocm_sysdeps;$SDK_ROOT/lib/llvm" "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK_ROOT --rocm-device-lib-path=$SDK_ROOT/lib/llvm/amdgcn/bitcode"
cmake --build "$BUILD_DIR" --target strata native_cache_test native_cache_eligibility_hip_test native_optional_save_test native_flag_hip_test conv_cache_test conversation_cache_test conversation_split_failure_test conversation_memory_test conversation_file_test message_boundary_test session_save_reclaim_test -j "${JOBS:-1}"
ctest --test-dir "$BUILD_DIR" --output-on-failure -R '^(native_cache|native_cache_eligibility_hip|native_optional_save|native_flag_hip|conv_cache|conversation_cache|conversation_split_failure|conversation_memory|conversation_file|message_boundary|session_save_reclaim)_test$'
