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
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(key, vars, _fun, _opts) do
    # The compiled function receives args_list: a list of flat thunk lists.
    # Each thunk is a zero-arity function that returns an actual tensor.
    # We bypass runtime_fun (which builds Expr trees) and instead:
    # 1. Call thunks to get actual tensors
    # 2. Transfer them to Mlx.Backend
    # 3. Reconstruct the container structure using vars as template
    # 4. Call the original function (key) directly with MLX-backed tensors
    # 5. Batch-evaluate all outputs
    fn args_list ->
      Enum.map(args_list, fn arg_funs ->
        prev = Nx.default_backend()

        try do
          Nx.default_backend(Mlx.Backend)

          # Call each thunk and transfer the resulting tensor to MLX
          mlx_tensors =
            Enum.map(arg_funs, fn thunk ->
              transfer_one(thunk.())
            end)

          # Reconstruct container structure from flat tensor list.
          # vars defines the nesting (maps, tuples, structs) while
          # mlx_tensors is the flat list of actual MLX-backed tensors.
          {mlx_args, []} =
            Enum.map_reduce(vars, mlx_tensors, fn var, remaining ->
              Nx.Defn.Composite.traverse(var, remaining, fn _expr, [tensor | rest] ->
                {tensor, rest}
              end)
            end)

          # Call the original function with MLX-backed tensors.
          # All Nx operations dispatch through Mlx.Backend, building
          # a lazy MLX computation graph without GPU execution.
          result = apply(key, mlx_args)

          # Batch-evaluate all output tensors — triggers MLX's graph
          # evaluation on Metal in one optimized pass.
          deep_eval(result)

          result
        after
          Nx.default_backend(prev)
        end
      end)
    end
  end

  @impl true
  def __partitions_options__(_opts), do: [groups: [default: :ok]]

  @impl true
  def __to_backend__(_opts), do: {Mlx.Backend, []}

  # --- Container traversal: transfer inputs to Mlx.Backend ---

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
