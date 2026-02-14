# ElixirMlx

An [Nx](https://github.com/elixir-nx/nx) backend and `Nx.Defn` compiler for
Apple's [MLX](https://github.com/ml-explore/mlx) machine learning framework,
targeting Apple Silicon.

ElixirMlx binds to MLX through [mlx-c](https://github.com/ml-explore/mlx-c),
the official stable C API, using Erlang NIFs.

## Why ElixirMlx?

[EMLX](https://github.com/elixir-nx/emlx) is the official Nx MLX backend
maintained by the elixir-nx team. It binds directly to MLX's C++ API.

ElixirMlx takes a different approach:

| | ElixirMlx | EMLX |
|---|---|---|
| **Binding layer** | mlx-c (C API) | Direct C++ |
| **ABI stability** | Stable across MLX releases (mlx-c is the official FFI surface, same as mlx-swift) | Breaks on MLX internal changes |
| **Build complexity** | C compiler only | C++ compiler + MLX headers |
| **Maintenance** | Tracks mlx-c releases | Tracks MLX internals |

mlx-c is the API that Apple designed for language bindings. mlx-swift uses it.
This project brings the same stability guarantees to Elixir.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4+)
- Erlang/OTP 26+
- Elixir 1.16+
- Xcode Command Line Tools (for Metal SDK)
- CMake (for building mlx-c)

## Installation

Add `elixir_mlx` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:elixir_mlx, "~> 0.1.0"}
  ]
end
```

Then fetch and compile:

```bash
mix deps.get
mix compile
```

The first compilation downloads and builds mlx-c from source. Subsequent
builds use the cached artifacts.

## Quick Start

### As an Nx Backend

```elixir
# Set as the default backend
Nx.default_backend(Mlx.Backend)

# All Nx operations now run on MLX (GPU by default)
a = Nx.tensor([1.0, 2.0, 3.0])
b = Nx.tensor([4.0, 5.0, 6.0])
Nx.add(a, b)
#=> #Nx.Tensor<
#     f32[3]
#     Mlx.Backend
#     [5.0, 7.0, 9.0]
#   >

# Or per-tensor
t = Nx.tensor([1, 2, 3], backend: Mlx.Backend)
```

### As a Defn Compiler

```elixir
defmodule MyModel do
  import Nx.Defn

  @defn_compiler Mlx.Compiler
  defn predict(x, w) do
    Nx.dot(x, w) |> Nx.sigmoid()
  end
end

x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
w = Nx.tensor([[0.5], [0.3]])
MyModel.predict(x, w)
```

### Device Management

MLX uses Apple Silicon's unified memory architecture. CPU and GPU share the
same memory space -- no data copies when switching devices.

```elixir
# GPU is the default on Apple Silicon
Mlx.Device.set_default(:gpu)

# Switch to CPU
Mlx.Device.set_default(:cpu)
```

### Explicit Evaluation

MLX operations are lazy by default. In Backend mode, results are
auto-evaluated for Nx compatibility. For explicit control:

```elixir
t = Nx.tensor([1.0, 2.0, 3.0])
Mlx.eval(t)          # Force evaluation
Mlx.synchronize()    # Wait for all queued GPU operations
```

## Supported Operations

ElixirMlx implements the `Nx.Backend` behaviour. Currently supported:

**Tensor creation:** `tensor`, `from_binary`, `eye`, `iota`, constants

**Element-wise unary:** `negate`, `abs`, `sign`, `ceil`, `floor`, `round`,
`exp`, `expm1`, `log`, `log1p`, `sqrt`, `rsqrt`, `sin`, `cos`, `tan`,
`asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`,
`erf`, `erfc`, `erf_inv`, `is_nan`, `is_infinity`, `logical_not`

**Element-wise binary:** `add`, `subtract`, `multiply`, `divide`, `quotient`,
`remainder`, `pow`, `atan2`, `equal`, `not_equal`, `less`, `less_equal`,
`greater`, `greater_equal`, `logical_and`, `logical_or`

**Reductions:** `sum`, `product`, `reduce_max`, `reduce_min`, `argmax`, `argmin`

**Shape:** `reshape`, `transpose`, `squeeze`, `broadcast`, `slice`,
`concatenate`, `as_type`

**Linear algebra:** `dot` (matrix multiply)

**Selection:** `select`

### Type Support

All MLX-supported types are mapped:

| Nx type | MLX type | Notes |
|---------|----------|-------|
| `{:f, 32}` | `float32` | Default float type |
| `{:f, 16}` | `float16` | |
| `{:bf, 16}` | `bfloat16` | |
| `{:s, 8..64}` | `int8..int64` | |
| `{:u, 8..64}` | `uint8..uint64` | |
| `{:c, 64}` | `complex64` | |
| `{:f, 64}` | -- | **Not supported** (Metal limitation) |

## Architecture

ElixirMlx uses a four-layer architecture:

```
Layer 4: Mlx module           -- High-level Elixir API (eval, synchronize)
Layer 3: Mlx.Backend          -- Nx.Backend behaviour (~35 callbacks)
         Mlx.Compiler         -- Nx.Defn.Compiler behaviour
Layer 2: Mlx.NIF, Mlx.Dtype   -- NIF declarations, type mapping
         Mlx.Device            -- Device management
Layer 1: c_src/mlx_nif.c      -- C NIF shim wrapping mlx-c calls
         mlx-c (pinned)       -- Official MLX C API
```

Key design decisions:

- **Resource-managed NIFs**: Every mlx-c object (array, stream, device) is
  wrapped in an Erlang NIF resource with a destructor. The BEAM's garbage
  collector triggers `mlx_*_free` calls automatically.
- **Thread-local error buffer**: mlx-c errors are captured per-scheduler-thread
  and surfaced as Elixir exceptions, keeping the BEAM safe.
- **Dirty schedulers**: Long-running operations (`eval`, `to_binary`,
  `synchronize`) run on dirty CPU schedulers to avoid blocking normal
  BEAM schedulers.
- **Metaprogrammed callbacks**: Unary and binary Nx.Backend callbacks are
  generated from mapping tables, keeping the implementation DRY.

## Development Status

ElixirMlx is in early development (Phase 1: core ops + Nx backend).

Planned phases:
1. Core array ops + Nx.Backend integration (current)
2. Lazy Nx.Defn.Compiler with MLX graph building
3. Neural network layers (`mlx.nn` equivalent)
4. Function transforms (grad, vmap, jit)
5. Model loading (safetensors / Hugging Face MLX format)

## Development

```bash
# Clone and build
git clone https://github.com/elixir-mlx/elixir_mlx
cd elixir_mlx
mix deps.get
mix compile    # Downloads + builds mlx-c on first run

# Run tests
mix test

# Format
mix format
```

## License

MIT License. See [LICENSE](LICENSE) for details.
