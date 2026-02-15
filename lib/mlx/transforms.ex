defmodule Mlx.Transforms do
  @moduledoc """
  MLX-native function transforms: grad, value_and_grad, vjp, jvp.

  These transforms operate on the MLX computation graph directly via mlx-c
  closures, enabling MLX-native automatic differentiation. This complements
  `Nx.Defn.grad` (which works symbolically through `Mlx.Compiler`) by
  providing direct access to MLX's transform machinery.

  ## Architecture: Closure Bridge

  Elixir functions can't be called directly from C function pointers. The
  closure bridge pattern solves this:

  1. A helper process is spawned that captures the Elixir function
  2. The NIF call runs on a dirty scheduler with the helper's PID
  3. When MLX calls the closure trampoline, it sends inputs to the helper
  4. The helper calls the Elixir function and responds via `closure_respond` NIF
  5. The trampoline unblocks and returns the result to MLX

  ## Usage

      # Value and gradient
      vag_fn = Mlx.Transforms.value_and_grad(fn [x] -> [Nx.sum(Nx.pow(x, 2))] end)
      {[value], [grad]} = vag_fn.([Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)])

      # Simple gradient
      grad_fn = Mlx.Transforms.grad(fn [x] -> [Nx.sum(Nx.pow(x, 2))] end)
      [grad] = grad_fn.([Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)])
  """

  @doc """
  Returns a function that computes gradients using MLX-native value_and_grad.

  The input function receives a list of Nx tensors and must return a list
  containing a single scalar tensor (the loss). `argnums` specifies which
  arguments to differentiate with respect to (0-indexed).

  ## Examples

      grad_fn = Mlx.Transforms.grad(fn [x] -> [Nx.sum(Nx.pow(x, 2))] end)
      [grad] = grad_fn.([Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)])
  """
  def grad(fun, argnums \\ [0]) do
    fn inputs ->
      {_values, grads} = do_value_and_grad(fun, inputs, argnums)
      grads
    end
  end

  @doc """
  Returns a function that computes both the function value and its gradients.

  Returns `{values, grads}` where both are lists of Nx tensors.
  """
  def value_and_grad(fun, argnums \\ [0]) do
    fn inputs ->
      do_value_and_grad(fun, inputs, argnums)
    end
  end

  @doc """
  Computes the vector-Jacobian product (VJP).

  `fun` takes a list of tensors, returns a list of tensors.
  `primals` and `cotangents` are lists of Nx tensors.

  Returns `{outputs, vjps}`.
  """
  def vjp(fun, primals, cotangents) do
    primals = Enum.map(primals, &ensure_mlx/1)
    cotangents = Enum.map(cotangents, &ensure_mlx/1)
    helper = spawn_helper(fun)

    primal_refs = Enum.map(primals, &extract_ref/1)
    cotangent_refs = Enum.map(cotangents, &extract_ref/1)

    case Mlx.NIF.vjp_apply(helper, primal_refs, cotangent_refs) do
      {:ok, {primal_list, vjp_list}} ->
        send(helper, :done)
        {wrap_refs(primal_list), wrap_refs(vjp_list)}

      {:error, reason} ->
        send(helper, :done)
        raise "MLX vjp failed: #{reason}"
    end
  end

  @doc """
  Computes the Jacobian-vector product (JVP).

  `fun` takes a list of tensors, returns a list of tensors.
  `primals` and `tangents` are lists of Nx tensors.

  Returns `{outputs, jvps}`.
  """
  def jvp(fun, primals, tangents) do
    primals = Enum.map(primals, &ensure_mlx/1)
    tangents = Enum.map(tangents, &ensure_mlx/1)
    helper = spawn_helper(fun)

    primal_refs = Enum.map(primals, &extract_ref/1)
    tangent_refs = Enum.map(tangents, &extract_ref/1)

    case Mlx.NIF.jvp_apply(helper, primal_refs, tangent_refs) do
      {:ok, {primal_list, tangent_list}} ->
        send(helper, :done)
        {wrap_refs(primal_list), wrap_refs(tangent_list)}

      {:error, reason} ->
        send(helper, :done)
        raise "MLX jvp failed: #{reason}"
    end
  end

  @doc """
  Returns a function that applies `fun` in a vectorized manner over the
  batch dimension specified by `in_axes` and `out_axes`.

  `fun` takes a list of tensors and returns a list of tensors.
  `in_axes` specifies which axis of each input to vectorize over (0-indexed).
  `out_axes` specifies which axis of each output to place the mapped dimension.

  ## Examples

      # Square each row in a batch
      vmap_fn = Mlx.Transforms.vmap(fn [x] -> [Nx.pow(x, 2)] end)
      result = vmap_fn.([Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Mlx.Backend)])
  """
  def vmap(fun, in_axes \\ [0], out_axes \\ [0]) do
    fn inputs ->
      inputs = Enum.map(inputs, &ensure_mlx/1)
      helper = spawn_helper(fun)
      input_refs = Enum.map(inputs, &extract_ref/1)

      case Mlx.NIF.vmap_apply(helper, input_refs, in_axes, out_axes) do
        {:ok, output_refs} ->
          send(helper, :done)
          wrap_refs(output_refs)

        {:error, reason} ->
          send(helper, :done)
          raise "MLX vmap failed: #{reason}"
      end
    end
  end

  # --- Private helpers ---

  defp do_value_and_grad(fun, inputs, argnums) do
    inputs = Enum.map(inputs, &ensure_mlx/1)
    helper = spawn_helper(fun)
    input_refs = Enum.map(inputs, &extract_ref/1)

    case Mlx.NIF.value_and_grad_apply(helper, input_refs, argnums) do
      {:ok, {value_list, grad_list}} ->
        send(helper, :done)
        {wrap_refs(value_list), wrap_refs(grad_list)}

      {:error, reason} ->
        send(helper, :done)
        raise "MLX value_and_grad failed: #{reason}"
    end
  end

  # Spawns a helper process that captures `fun` and handles trampoline
  # callbacks from the C closure bridge.
  defp spawn_helper(fun) do
    spawn(fn -> helper_loop(fun) end)
  end

  defp helper_loop(fun) do
    receive do
      {:trampoline_call, bridge_ref, input_refs} ->
        # Convert raw array refs to Nx tensors
        inputs = Enum.map(input_refs, &ref_to_tensor/1)

        try do
          results = fun.(inputs)
          results = if is_list(results), do: results, else: [results]

          result_refs =
            Enum.map(results, fn
              %Nx.Tensor{data: %Mlx.Backend{ref: ref}} -> ref
              tensor -> extract_ref(ensure_mlx(tensor))
            end)

          Mlx.NIF.closure_respond(bridge_ref, result_refs)
        rescue
          _e ->
            Mlx.NIF.closure_respond_error(bridge_ref)
        end

        helper_loop(fun)

      :done ->
        :ok

    after
      30_000 -> :ok
    end
  end

  defp ref_to_tensor(ref) do
    shape =
      case Mlx.NIF.shape(ref) do
        {:ok, s} -> List.to_tuple(s)
        _ -> {}
      end

    type =
      case Mlx.NIF.dtype(ref) do
        {:ok, dtype_atom} -> Mlx.Dtype.to_nx(dtype_atom)
        _ -> {:f, 32}
      end

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      shape: shape,
      type: type,
      names: List.duplicate(nil, tuple_size(shape))
    }
  end

  defp ensure_mlx(%Nx.Tensor{data: %Mlx.Backend{}} = t), do: t
  defp ensure_mlx(%Nx.Tensor{} = t), do: Nx.backend_transfer(t, Mlx.Backend)

  defp extract_ref(%Nx.Tensor{data: %Mlx.Backend{ref: ref}}), do: ref

  defp wrap_refs(refs) when is_list(refs) do
    Enum.map(refs, &ref_to_tensor/1)
  end
end
