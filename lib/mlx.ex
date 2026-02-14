defmodule Mlx do
  @moduledoc """
  MLX bindings for Elixir via mlx-c.
  Satisfies: B1 (feature-complete), U2 (Elixir-idiomatic API)

  An Nx backend and Nx.Defn compiler for Apple's MLX machine learning
  framework on Apple Silicon. Uses the official mlx-c API for ABI stability.

  ## Quick Start

      # Use as Nx backend
      Nx.default_backend(Mlx.Backend)
      t = Nx.tensor([1.0, 2.0, 3.0])
      Nx.sum(t)

      # Or per-tensor
      Nx.tensor([1, 2, 3], backend: Mlx.Backend)

      # Use as defn compiler
      defmodule MyModel do
        import Nx.Defn
        @defn_compiler Mlx.Compiler
        defn predict(x, w), do: Nx.dot(x, w)
      end

  ## Device Management

      Mlx.Device.set_default(:gpu)  # GPU is default on Apple Silicon
      Mlx.Device.set_default(:cpu)
  """

  @doc """
  Evaluates a tensor, forcing all pending lazy computations.

  In normal Nx usage this is called automatically. Use this when you
  want explicit control over when GPU computation happens.
  """
  def eval(%Nx.Tensor{data: %Mlx.Backend{ref: ref}} = tensor) do
    case Mlx.NIF.eval(ref) do
      :ok -> tensor
      {:error, reason} -> raise "MLX eval failed: #{reason}"
    end
  end

  @doc """
  Synchronizes the given stream, waiting for all queued operations to complete.
  """
  def synchronize(stream \\ nil) do
    stream = stream || Mlx.Backend.default_stream()

    case Mlx.NIF.synchronize(stream) do
      :ok -> :ok
      {:error, reason} -> raise "MLX synchronize failed: #{reason}"
    end
  end
end
