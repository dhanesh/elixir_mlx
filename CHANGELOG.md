# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-02-16

### New Modules

- **Mlx.Transforms** — MLX-native function transforms: `grad`, `value_and_grad`, `vjp`, `jvp`, `vmap`
  via closure bridge pattern (C trampoline + Elixir helper process)
- **Mlx.NN** — Functional neural network operations: 13 activations (relu, gelu, silu, etc.),
  5 loss functions (mse, cross_entropy, etc.), functional layers (linear, embedding, dropout),
  re-exports from Mlx.Fast (layer_norm, rms_norm, sdpa, rope)
- **Mlx.Hub** — HuggingFace Hub integration: download, snapshot_download, list_repo_files,
  cached_path with local disk caching (requires optional `req` dep)
- **Mlx.Models** — High-level model loading pipeline: load_model_state (download → load →
  dequantize → unflatten → Axon.ModelState), load_config, unflatten_params, to_model_state

### Enhancements

- **mlx-c upgraded to v0.5.0** (targets MLX v0.30.6, was v0.1.2)
- **Mlx.IO.load_weights/1,2** — Unified weight loader with format auto-detection (.npy,
  .safetensors, directories), sharded safetensors support, optional dtype casting
- Python comparison test suite (44 tests vs pre-generated Python MLX reference values)
- Error quality tests (11 tests asserting error message content)

### Dependencies

- `jason` promoted from test-only to runtime dependency
- `req ~> 0.5` added as optional dependency (for Mlx.Hub)
- `axon ~> 0.8` and `polaris ~> 0.1` changed from test-only to optional

### Stats

- 469 tests, 0 failures (up from 350 in v0.1.0)
- ~136 NIFs in c_src/mlx_nif.c

## [0.1.0] - 2026-02-16

Initial release of ElixirMlx — MLX bindings for Elixir via mlx-c.

### Nx.Backend

- 72 Nx.Backend callbacks implemented (66 native MLX, 6 via BinaryBackend fallback)
- Full type support: f32, f16, bf16, s8-s64, u8-u64, c64 (f64 excluded — Metal limitation)
- Lazy evaluation with auto-eval for Nx compatibility
- Resource-managed NIFs with automatic cleanup via BEAM GC
- Thread-local error buffer for BEAM scheduler safety
- Dirty schedulers for `eval`, `to_binary`, `synchronize`

#### Tensor Operations

- **Creation**: `tensor`, `from_binary`, `eye`, `iota`, constants
- **Unary**: `negate`, `abs`, `sign`, `ceil`, `floor`, `round`, `exp`, `expm1`, `log`,
  `log1p`, `sqrt`, `rsqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`,
  `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `erfc`, `erf_inv`, `cbrt`, `is_nan`,
  `is_infinity`, `logical_not`, `bitwise_not`, `conjugate`, `real`, `imag`
- **Binary**: `add`, `subtract`, `multiply`, `divide`, `quotient`, `remainder`, `pow`,
  `atan2`, `min`, `max`, `equal`, `not_equal`, `less`, `less_equal`, `greater`,
  `greater_equal`, `logical_and`, `logical_or`, `logical_xor`, `bitwise_and`,
  `bitwise_or`, `bitwise_xor`, `left_shift`, `right_shift`
- **Reductions**: `sum`, `product`, `reduce_max`, `reduce_min`, `argmax`, `argmin`,
  `any`, `all`, `cumulative_sum`, `cumulative_product`, `cumulative_max`, `cumulative_min`
- **Shape**: `reshape`, `transpose`, `squeeze`, `broadcast`, `slice`, `put_slice`,
  `concatenate`, `stack`, `as_type`, `bitcast`, `pad`, `reverse`, `take`,
  `gather`, `indexed_add`, `indexed_put`
- **Linear algebra**: `dot`, `conv`, `triangular_solve`, `lu`, `qr`, `svd`, `cholesky`,
  `eigh`, `solve`
- **Selection**: `select`, `where`, `sort`, `argsort`, `to_batched`, `clip`
- **Window**: `window_sum`, `window_max`, `window_min`, `window_product`
  (via `as_strided` + reductions)
- **FFT**: `fft`, `ifft`
- **Fallback ops** (via BinaryBackend): `count_leading_zeros`, `population_count`,
  `reduce`, `window_reduce`, `window_scatter_max`, `window_scatter_min`

### Nx.Defn Compiler

- `Mlx.Compiler` implements `Nx.Defn.Compiler` for lazy batch evaluation
- MLX computation graph built during `defn`, evaluated once at the end
- Compatible with Nx 0.10 calling conventions

### Domain Modules

- **Mlx.Random** — key generation, uniform, normal, bernoulli, randint, categorical,
  truncated_normal, gumbel, laplace, permutation
- **Mlx.Linalg** — inv, pinv, norm, cholesky, qr, svd, eigh, eigvalsh, lu, solve,
  triangular_solve, cross
- **Mlx.FFT** — fft, ifft, fft2, fftn, rfft, irfft, rfft2, rfftn
- **Mlx.Fast** — layer_norm, rms_norm, rope, scaled_dot_product_attention
- **Mlx.Quantize** — quantize, dequantize, quantized_matmul
- **Mlx.IO** — save/load arrays and safetensors format

### Device Management

- Unified memory architecture (CPU/GPU share memory, no copies)
- `Mlx.Device.set_default/1` for device switching
- GPU default on Apple Silicon

### Build System

- cc_precompiler configured for `aarch64-apple-darwin`
- NIF versions 2.16 and 2.17
- CI workflow for macOS Apple Silicon

### Ecosystem Integration

- Verified with Axon (model predict + training)
- Verified with Scholar (linear regression, KMeans clustering)
- Verified with Polaris optimizers

### Known Limitations

- f64 not supported (Metal limitation)
- `from_pointer`/`to_pointer` not supported (use `Nx.from_binary`/`Nx.to_binary`)
- Function transforms (grad, vmap) require mlx-c closure support (added in v0.2.0)

[0.2.0]: https://github.com/elixir-mlx/elixir_mlx/releases/tag/v0.2.0
[0.1.0]: https://github.com/elixir-mlx/elixir_mlx/releases/tag/v0.1.0
