# mikenavi Strata application and release source

Application0.2.6 and the native engine are independent components in the same
upstream project. Both source branches include upstream4b9e05a2. The custom native
CMake project is0.1.51; the unchanged upstream Qwen package version is0.1.50. mikenavi/stable contains the application with stable watcher
birth stamps and native probability/completion/tokenizer APIs. mikenavi/engine
contains the native cache and probability changes. local/components.lock.json
pins an exact engine commit; the release assembler never applies loose patches.

Clone this repository with its branches. If using a single-branch clone, fetch
the exact engine_commit from the lock before assembly. Run:

```sh
python3 local/build-release.py --work-dir /path/to/new-build --prepare-only
SDK_ROOT=/path/to/pinned/rocm GGML_SOURCE=/path/to/pinned/ggml \
  python3 local/build-release.py --work-dir /path/to/another-build
```

The first command checks source identities and exports application/native source
from committed Git objects. The second builds and links the native engine and
runs its CPU cache/ownership tests, with JOBS=1 by default. Supply the pinned
ggml_revision and ROCm SDK; weights, signed vision packages and runtime/container
integration are separate deployment inputs. No key, host IP, weight or cache is
stored here. The stock upstream installer uses its own engine feed; fleet
containers must select the custom native binary explicitly.

Read release.json to identify both source commits. A combined app version is
not an engine version. Keep custom native artifacts distinct from signed official
packages. Application CPU tests and native source checks were passed. The native upstream
sync changes one Qwen generate.cpp comment and the CMake version; a fresh linked
HIP build and GPU/workload checks remain pending. Source sync does not activate
a runtime or qualify DeepSeek, Metal or SYCL changes.
Merge/rebase upstream per component, run regressions and update the lock before
tagging another fleet release. Branch names are development conveniences;
builds and deployments use exact commits and image digests.
