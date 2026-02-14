# mlx-elixir

## Outcome

Create a production-quality MLX binding for Elixir that is simple to use, has minimal external dependencies, is robust with close to 100% test coverage, and is feature-complete compared to the Python (`mlx`) and Swift (`mlx-swift`) alternatives.

---

## Constraints

### Business

#### B1: Full Feature Parity with Model Loading

Achieve full API parity with mlx-python including: core array operations, neural network layers (mlx.nn), optimizers, function transforms (grad, vmap, jit), random number generation, FFT, and the ability to load/run models from Hugging Face MLX format (safetensors).

> **Rationale:** The outcome statement requires feature-completeness compared to Python and Swift alternatives. Model loading is essential for practical ML workflows on Apple Silicon.

#### B2: Nx Backend Compatibility

Must implement the `Nx.Backend` behaviour so that elixir_mlx tensors integrate seamlessly with the entire Nx ecosystem (Axon, Scholar, Bumblebee, etc.).

> **Rationale:** The Elixir ML ecosystem is built around Nx. Without Nx compatibility, elixir_mlx would be an island — users couldn't leverage existing models, training loops, or data pipelines. This is how EXLA and Torchx integrate.

#### B3: Production-Quality Open Source Package

The library must be production-quality: well-documented, well-tested, published on Hex.pm, with clear versioning and a stable public API.

> **Rationale:** The outcome statement explicitly requires "production-quality." An MLX binding only has value if the Elixir community can depend on it.

---

### Technical

#### T1: NIF Binding via mlx-c

Interface with MLX through Erlang NIFs binding to the official [mlx-c](https://github.com/ml-explore/mlx-c) C API. This is the same strategy used by mlx-swift.

> **Rationale:** mlx-c is the official, maintained C API for MLX. NIF binding provides the lowest-overhead integration with the BEAM VM. Using the official C API ensures we track upstream MLX releases.

#### T2: Defensive NIF Safety Model

All NIFs must use resource objects with reference counting, dirty schedulers for operations > 1ms, and yielding NIFs for very long operations. A crash in MLX C code must never take down the BEAM VM.

> **Rationale:** The BEAM VM's reliability model depends on process isolation. A segfault in a NIF kills the entire VM. Defensive NIFs with proper resource management protect the BEAM's fault tolerance guarantees.

#### T3: macOS Apple Silicon Only

The library targets macOS on Apple Silicon (M1/M2/M3/M4/M5+) exclusively. No Linux, no Intel Mac support.

> **Rationale:** MLX is designed exclusively for Apple Silicon's unified memory architecture. There is no cross-platform abstraction — the hardware IS the feature.

#### T4: Correctness Over Speed

Prioritize correct, safe implementation over performance optimization. Optimize based on profiling data after correctness is established.

> **Rationale:** Premature optimization in NIF code is dangerous — it can introduce memory corruption, race conditions, and BEAM scheduler starvation. Get it right first, then make it fast.

#### T5: Minimal External Dependencies

Beyond mlx-c itself, the library should have minimal external C/C++ dependencies. Elixir-side dependencies should be limited to Nx and standard build tools.

> **Rationale:** The outcome statement requires "minimal external dependencies." Each dependency is a potential build failure, version conflict, or security surface.

#### T6: Preserve MLX Lazy Evaluation

MLX's lazy evaluation model must be faithfully exposed in Elixir. Operations build a computation graph; `eval()` triggers execution.

> **Rationale:** Lazy evaluation is a core MLX design principle that enables graph optimization, automatic differentiation, and efficient memory use. Eagerly evaluating would defeat MLX's performance model.

#### T7: Leverage Unified Memory Model

Must expose MLX's unified memory (CPU+GPU access without data transfers) through the API. Users should be able to target CPU or GPU streams.

> **Rationale:** Unified memory is Apple Silicon's killer feature for ML. The binding must make this accessible, not hide it behind abstractions.

---

### User Experience

#### U1: Elixir-Idiomatic API Design

The public API must follow Elixir conventions: pipe-friendly functions, pattern matching on results, protocols for extensibility, behaviours for contracts. Follow Nx naming conventions where applicable.

> **Rationale:** Elixir developers expect `tensor |> Mlx.add(other) |> Mlx.eval()`, not `Mlx.eval(Mlx.add(tensor, other))`. An un-idiomatic API will see no adoption regardless of technical merit.

#### U2: Nx Ecosystem Interoperability

Tensors created by elixir_mlx should work transparently with Axon (neural networks), Scholar (classical ML), Explorer (dataframes), and other Nx-compatible libraries.

> **Rationale:** The value of Nx backend compatibility (B2) is only realized if interop actually works end-to-end with the ecosystem.

#### U3: Clear Error Messages from NIF Layer

MLX errors, invalid tensor shapes, type mismatches, and device errors must be translated into clear Elixir exceptions with actionable messages — not cryptic NIF crash dumps.

> **Rationale:** NIF errors are notoriously opaque. Good error messages are the difference between a usable library and one that developers abandon after the first bug.

---

### Security

#### S1: NIF Crashes Must Not Take Down BEAM

Under no circumstances should a bug or invalid input in the NIF layer cause the BEAM VM to crash. All NIF entry points must validate inputs and handle errors gracefully.

> **Rationale:** BEAM applications often run multiple services in a single VM. A NIF segfault kills everything — not just the ML operation, but potentially the entire production system.

#### S2: No Memory Leaks from NIF Resources

All MLX arrays, streams, and other C resources must be properly reference-counted and freed when their Elixir references are garbage collected. Resource destructors must be registered.

> **Rationale:** Memory leaks in NIFs are invisible to Erlang's process-level memory tracking. They accumulate silently until the OS kills the process. Proper resource management is non-negotiable.

---

### Operational

#### O1: High Test Coverage with Integration Tests

Target 90%+ unit test coverage plus integration tests that run actual MLX operations on Apple Silicon hardware. Include numerical accuracy tests comparing results against Python MLX output.

> **Rationale:** The outcome requires "close to 100% test coverage." Integration tests on real hardware catch issues that mocked tests miss — especially for NIF code where behavior depends on the hardware.

#### O2: CI on Apple Silicon Runners

Continuous integration must run on macOS Apple Silicon runners (GitHub Actions M1/M2 runners or self-hosted). Linux CI can only verify compilation stubs, not runtime behavior.

> **Rationale:** MLX only works on Apple Silicon. CI that doesn't run on Apple Silicon can't catch real bugs. This is a hard platform constraint.

#### O3: Hex.pm Publication

The package must be publishable and installable via `mix deps.get` from Hex.pm with precompiled NIF binaries for supported macOS/Apple Silicon targets.

> **Rationale:** Requiring users to compile mlx-c from source would dramatically reduce adoption. Precompiled binaries (via `elixir_make` or `rustler_precompiled` patterns) are the standard for NIF packages.

---

## Tensions

### TN1: Nx Eager Backend vs MLX Lazy Evaluation

**Between:** B2 (Nx Backend Compatibility) ↔ T6 (Preserve MLX Lazy Evaluation)

`Nx.Backend` expects eager tensor operations — each operation returns a fully evaluated tensor. MLX is fundamentally lazy — operations build a computation graph that only executes on `eval()`. Implementing `Nx.Backend` naively would force eager evaluation on every operation, destroying MLX's graph optimization, automatic differentiation, and memory efficiency.

**Resolution options considered:**
- A. **Both Backend + Compiler** — Implement `Nx.Backend` (auto-eval after each op) AND `Nx.Defn.Compiler` (preserve laziness in `defn` blocks). This is the EXLA pattern.
- B. Backend-only with auto-eval — Simple but sacrifices MLX's core value proposition.
- C. Compiler-only — Preserves laziness but breaks eager Nx code outside `defn`.

> **Resolution:** Option A — Dual integration. The `Nx.Backend` auto-evaluates for compatibility with eager Nx code. The `Nx.Defn.Compiler` preserves full lazy evaluation within `defn` blocks, where performance-critical ML code lives. This is exactly how EXLA integrates: `EXLA.Backend` for eager ops, `EXLA` compiler for JIT within `defn`.

### TN2: Full Feature Parity vs Correctness-First

**Between:** B1 (Full Feature Parity) ↔ T4 (Correctness Over Speed)

mlx-python exposes hundreds of functions across `mlx.core`, `mlx.nn`, `mlx.optimizers`, and transforms (`grad`, `vmap`, `jit`). Implementing all of them with a correctness-first approach (T4) means each function gets thorough testing, defensive NIF wrapping, and Elixir-idiomatic API design. The scope is enormous — correctness-first and full parity are in direct tension on timeline.

> **Resolution:** Phased delivery. Each phase reaches full correctness before the next begins:
> 1. Core array ops + math → 2. Nx.Backend integration → 3. Neural network layers → 4. Function transforms → 5. Model loading (safetensors)
>
> This satisfies both constraints: full parity is the destination (B1), correctness is the pace (T4).

### TN3: Elixir-Idiomatic API vs Python Example Portability

**Between:** U1 (Elixir-Idiomatic API) ↔ B1 (Full Feature Parity)

Full parity with Python MLX means users will frequently port Python examples and tutorials. If the Elixir API diverges too much from Python naming, porting becomes a constant lookup exercise. But if the API mirrors Python too closely (`mlx_core_add(a, b)` style), it won't feel natural to Elixir developers.

> **Resolution:** Follow Nx naming conventions, which are already Elixir-idiomatic AND familiar to the ML community. Where MLX has unique operations not in Nx, use Elixir naming conventions. Provide a Python-to-Elixir function mapping table in documentation. Elixir idioms (pipes, pattern matching) always win where they improve the API.

### TN4: Minimal Dependencies vs Precompiled Binary Infrastructure

**Between:** T5 (Minimal External Dependencies) ↔ O3 (Hex.pm Publication)

Publishing precompiled NIF binaries requires build infrastructure: `elixir_make` or `cc_precompiler`, a CI build matrix for macOS ARM64 targets, and the mlx-c C++ library itself (which pulls in the Metal framework). This conflicts with the "minimal dependencies" constraint.

> **Resolution:** Accept mlx-c + Metal as unavoidable platform dependencies (they ARE the binding). Use `elixir_make` as the single build-tooling dependency. Precompiled binaries eliminate build-time dependencies for end users — from their perspective, `mix deps.get` just works. The "minimal" constraint applies to what users must install, not what the build system uses.

### TN5: BEAM Safety vs Full Feature Coverage

**Between:** S1 (NIF Crashes Must Not Take Down BEAM) ↔ B1 (Full Feature Parity)

Some MLX features are inherently unsafe to expose through NIFs: custom Metal kernel compilation, raw memory operations, and low-level compilation transforms can produce segfaults that no amount of NIF wrapping can prevent. Guaranteeing BEAM safety (S1, invariant) while exposing everything (B1, goal) creates a direct conflict.

> **Resolution:** Tiered safety model. The safe core (array ops, math, neural network layers, optimizers, standard transforms) is fully exposed with defensive NIF wrapping. Unsafe features (custom Metal kernels, raw memory operations) are gated behind an explicit opt-in module (`Mlx.Unsafe` or similar) with clear documentation about BEAM crash risks. S1 (invariant) takes precedence over B1 (goal) — we never claim the unsafe module is BEAM-safe.

### TN6: mlx-c Version Tracking vs Precompiled Binary Stability

**Between:** T1 (NIF via mlx-c) ↔ O3 (Hex.pm Publication)

T1 requires tracking upstream mlx-c releases to stay current with MLX features. But O3 requires stable precompiled binaries on Hex.pm. Each mlx-c update invalidates all precompiled binaries. Frequent upstream tracking creates a binary rebuild and re-publish burden.

> **Resolution:** Pin mlx-c to specific versions per elixir_mlx release. Use semantic versioning alignment: mlx-c minor version bumps correspond to elixir_mlx minor version bumps. Precompiled binaries are built per pinned version. Users get stability within a version; they get new MLX features by upgrading elixir_mlx versions.

---

## Required Truths

**Backward reasoning from outcome:** "For a production-quality, feature-complete MLX binding for Elixir to exist, what MUST be true?"

**Context:** [EMLX](https://github.com/elixir-nx/emlx) (v0.2.0) already exists as the official elixir-nx MLX backend using direct C++ bindings. This project differentiates by using [mlx-c](https://github.com/ml-explore/mlx-c), the official stable C API designed for language bindings (same approach as mlx-swift).

### RT-1: Project Is Differentiated from EMLX via mlx-c

The project must clearly articulate why it exists alongside EMLX. The key differentiator: mlx-c is the official, stable C API designed for language bindings (used by mlx-swift), offering better ABI stability than EMLX's direct C++ approach. This must be stated in README, docs, and reflected in the architecture.

**Gap:** No project scaffold, README, or documentation exists yet.

### RT-2: mlx-c Compiles and Links on macOS ARM64

mlx-c must be buildable as a static or shared library and linkable into an Erlang NIF `.so` on macOS ARM64. This requires: CMake, Xcode CLI tools, Metal SDK, and integration with `elixir_make`.

**Gap:** Build pipeline not established. Need to verify: (1) mlx-c can produce a static lib, (2) the static lib can link into a NIF without symbol conflicts, (3) the Metal framework links correctly.

### RT-3: Every mlx-c Object Type Has a NIF Resource with Destructor

mlx-c exposes these object types that hold C-side memory: `array`, `stream`, `device`, `string`, `closure`, `distributed_group`, `map`, `vector`, `optional`. Each must have a corresponding `enif_open_resource_type` with a `dtor` callback that calls the appropriate `mlx_*_free` function.

**Gap:** No resource types defined. Need to enumerate the full set of mlx-c types requiring GC-safe wrappers.

### RT-4: mlx-c Error Handler Routes Errors to Elixir Exceptions

mlx-c uses a callback-based error model (`mlx_set_error_handler`). A custom handler must be registered at NIF load time that captures error messages and converts them into `enif_raise_exception` calls with descriptive Elixir error terms.

**Gap:** Error handler not implemented. Need to verify: (1) handler is called synchronously (safe for NIF context), (2) error messages include sufficient context (file, line, operation), (3) handler works correctly from dirty scheduler threads.

### RT-5: Nx.Backend Behaviour Callbacks Are Fully Implemented

All required callbacks in `Nx.Backend` / `Nx.Tensor` behaviour must be implemented, mapping each Nx operation to its corresponding mlx-c function. Each callback must auto-evaluate (call `mlx_eval`) before returning, per TN1 resolution.

**Gap:** No backend implementation exists. Need to: (1) enumerate all Nx.Backend callbacks (~100+ operations), (2) map each to mlx-c functions, (3) identify any Nx operations without mlx-c equivalents.

### RT-6: Nx.Defn.Compiler Implements All 4 Required Callbacks

Must implement `__jit__/5`, `__compile__/4`, `__partitions_options__/1`, and `__to_backend__/1`. The compiler must receive defn expression graphs, translate them to MLX lazy operations (preserving T6), and execute them on the MLX runtime.

**Gap:** No compiler implementation exists. Need to understand: (1) how the `fun` parameter in `__jit__` builds the expression graph, (2) how to translate Nx expressions to MLX operations without eager evaluation, (3) caching strategy for compiled functions.

### RT-7: MLX Dtypes Map Bidirectionally to Nx Dtypes

Every Nx dtype must have a corresponding MLX dtype mapping, with one critical exception: **Metal does not support f64 (64-bit floats)**. This limitation must be documented, and `Nx.type/1` queries must report supported types accurately.

**Gap:** Dtype mapping table not defined. Known gaps: f64 unsupported, c128 (complex128) likely unsupported. Need exhaustive mapping: `{bf16, f16, f32} × {s8, s16, s32, s64} × {u8, u16, u32, u64} × {c64}`.

### RT-8: Dirty Scheduler Integration Covers All Long-Running Operations

All NIF functions that call mlx-c compute operations (matmul, conv, eval, compile) must be registered with `ERL_NIF_DIRTY_JOB_CPU_BOUND` flags. Simple accessor NIFs (shape, dtype, device queries) can run on normal schedulers.

**Gap:** No scheduler strategy defined. Need to: (1) categorize mlx-c functions by expected latency, (2) benchmark representative operations, (3) decide dirty vs normal for each NIF.

### RT-9: Precompiled NIF Binaries Can Be Built and Distributed

The build system must produce self-contained NIF binaries (`.so` files with mlx-c statically linked) distributable via Hex.pm. Users should not need CMake, Xcode CLI tools, or mlx-c source to use the package.

**Gap:** Build pipeline not established. Need to evaluate: (1) `elixir_make` with `cc_precompiler` for precompiled binaries, (2) static vs dynamic linking of mlx-c, (3) EMLX's approach via `cocoa-xu/mlx-build` as reference.

### RT-10: Apple Silicon CI Pipeline Is Configured

GitHub Actions with `macos-14` (M1) or `macos-15` (M3/M4) runners must be configured for: (1) building mlx-c + NIF from source, (2) running ExUnit integration tests, (3) building precompiled binaries for release.

**Gap:** No CI configuration exists. Need to verify: (1) GitHub Actions Apple Silicon runner availability and pricing, (2) build times for mlx-c compilation, (3) caching strategy for mlx-c builds.

---

## Solution Space

### Option A: Bottom-Up Layered Architecture (Recommended)

```
┌─────────────────────────────────────────┐
│  Layer 4: High-Level Elixir API         │  Mlx module
│  (pipes, protocols, Elixir-idiomatic)   │  User-facing
├─────────────────────────────────────────┤
│  Layer 3: Nx Integration               │  Mlx.Backend
│  (Nx.Backend + Nx.Defn.Compiler)       │  Mlx.Compiler
├─────────────────────────────────────────┤
│  Layer 2: Resource Management           │  Mlx.Native
│  (NIF resources, error handling,        │  Internal
│   dirty schedulers, dtype mapping)      │
├─────────────────────────────────────────┤
│  Layer 1: Raw C NIF Shim               │  c_src/mlx_nif.c
│  (thin wrapper around mlx-c calls)     │  Private
├─────────────────────────────────────────┤
│  mlx-c (pinned version, statically     │  External
│  linked)                               │  dependency
└─────────────────────────────────────────┘
```

- **Satisfies:** All 10 required truths
- **Build order:** L1 → L2 → L3 → L4 (matches phased delivery from TN2)
- **Testability:** Each layer independently testable
- **Complexity:** High but well-structured

### Option B: Nx.Backend-First (Top-Down)

Start by implementing `Nx.Backend` callbacks, building NIF bindings as needed per operation.

- **Satisfies:** RT-5, RT-6 fastest
- **Gap:** No clean layer separation; NIF safety (RT-3, RT-4, RT-8) mixed with business logic
- **Complexity:** Medium, but harder to refactor later

### Option C: Code-Generated Bindings

Parse mlx-c header files to auto-generate NIF function stubs, resource types, and dirty scheduler flags.

- **Satisfies:** RT-3, RT-4, RT-8 automatically
- **Gap:** Generator quality risk; edge cases in C header parsing; generated code harder to debug
- **Complexity:** Very high upfront, low ongoing (if generator works)

### Recommendation: Option A

Option A maps cleanly to the constraint structure: Layer 1 satisfies T1, Layer 2 satisfies T2/S1/S2/U3, Layer 3 satisfies B2/T6, Layer 4 satisfies U1. The phased delivery (TN2) follows the layer stack naturally. Each layer can be tested independently before the next is built on top.
