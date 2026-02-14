defmodule Mlx.Compiler do
  @moduledoc """
  Nx.Defn.Compiler implementation for MLX.
  Satisfies: RT-6 (Nx.Defn.Compiler), T6 (lazy evaluation)

  Preserves MLX's lazy evaluation model within `defn` blocks. When a
  `defn` function is compiled with `Mlx.Compiler`, operations build an
  MLX computation graph without evaluating. The entire graph is
  batch-evaluated once at the end, allowing MLX to optimize Metal
  kernel launches across the full computation.

  This follows the same pattern as EXLA: the compiler sets the backend,
  transfers inputs, runs the defn function (building a lazy graph),
  and batch-evaluates all outputs.

  ## Usage

      defmodule MyModel do
        import Nx.Defn

        @defn_compiler Mlx.Compiler
        defn predict(x, w) do
          Nx.dot(x, w) |> Nx.sigmoid()
        end
      end

  ## How it works

  1. Save the current default backend
  2. Set `Mlx.Backend` as default
  3. Deep-transfer all inputs to `Mlx.Backend`
  4. Run the `defn` function — all Nx ops dispatch through `Mlx.Backend`,
     which calls lazy mlx-c operations (e.g. `mlx_add`, `mlx_matmul`).
     No GPU execution happens yet.
  5. Batch-evaluate all output tensors — this triggers MLX's graph
     evaluation, executing the entire computation on Metal in one
     optimized pass.
  6. Restore the previous default backend
  """

  @behaviour Nx.Defn.Compiler

  @impl true
  def __jit__(key, vars, fun, args, opts) do
    __compile__(key, vars, fun, opts).(args)
  end

  @impl true
  def __compile__(_key, _vars, fun, _opts) do
    fn args ->
      # Save current default backend so we can restore it
      prev = Nx.default_backend()

      try do
        # Set MLX as the default backend for this computation
        Nx.default_backend(Mlx.Backend)

        # Transfer all inputs to Mlx.Backend. If already on Mlx.Backend,
        # this is a no-op. For tensors on other backends, this converts
        # them to lazy MLX arrays.
        mlx_args = deep_transfer(args)

        # Run the defn function. All Nx operations dispatch through
        # Mlx.Backend, which calls lazy mlx-c operations. Each NIF call
        # (mlx_add, mlx_matmul, etc.) returns immediately with a handle
        # to a lazy array node in the computation graph. No Metal
        # execution happens here.
        result = fun.(mlx_args)

        # Batch-evaluate all output tensors. This triggers MLX's graph
        # evaluation — MLX traces from outputs backward through the
        # computation graph and launches optimized Metal kernels.
        # After this, all output tensors contain computed values.
        deep_eval(result)

        result
      after
        # Always restore the previous default backend
        Nx.default_backend(prev)
      end
    end
  end

  @impl true
  def __partitions_options__(_opts), do: [groups: [default: :ok]]

  @impl true
  def __to_backend__(_opts), do: {Mlx.Backend, []}

  # --- Container traversal: transfer inputs to Mlx.Backend ---

  defp deep_transfer(args) when is_list(args) do
    Enum.map(args, &transfer_one/1)
  end

  defp transfer_one(%Nx.Tensor{data: %Mlx.Backend{}} = tensor), do: tensor

  defp transfer_one(%Nx.Tensor{} = tensor) do
    Nx.backend_transfer(tensor, Mlx.Backend)
  end

  defp transfer_one(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&transfer_one/1)
    |> List.to_tuple()
  end

  defp transfer_one(%{__struct__: _} = struct) do
    struct
    |> Map.from_struct()
    |> Map.new(fn {k, v} -> {k, transfer_one(v)} end)
    |> then(&struct(struct.__struct__, &1))
  end

  defp transfer_one(%{} = map) do
    Map.new(map, fn {k, v} -> {k, transfer_one(v)} end)
  end

  defp transfer_one(other), do: other

  # --- Container traversal: batch-evaluate all output tensors ---

  defp deep_eval(%Nx.Tensor{data: %Mlx.Backend{ref: ref}}) do
    case Mlx.NIF.eval(ref) do
      :ok -> :ok
      {:error, reason} -> raise "MLX eval failed: #{reason}"
    end
  end

  defp deep_eval(%Nx.Tensor{} = _tensor) do
    # Tensor on a different backend — nothing to eval
    :ok
  end

  defp deep_eval(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.each(&deep_eval/1)
  end

  defp deep_eval(list) when is_list(list) do
    Enum.each(list, &deep_eval/1)
  end

  defp deep_eval(%{__struct__: _} = struct) do
    struct
    |> Map.from_struct()
    |> Enum.each(fn {_k, v} -> deep_eval(v) end)
  end

  defp deep_eval(%{} = map) do
    Enum.each(map, fn {_k, v} -> deep_eval(v) end)
  end

  defp deep_eval(_other), do: :ok
end
