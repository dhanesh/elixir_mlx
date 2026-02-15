defmodule Mlx.Backend do
  @moduledoc """
  Nx.Backend implementation for Apple MLX via mlx-c.
  Satisfies: T2 (Nx compatibility), RT-5 (Nx.Backend), U1 (Nx.tensor works)

  Usage:
      Nx.tensor([1, 2, 3], backend: Mlx.Backend)
      Nx.default_backend(Mlx.Backend)

  MLX operations are lazy by default. The computation graph is only
  evaluated when data is needed (to_binary, inspect). In Backend mode,
  we auto-evaluate for Nx compatibility (TN1 resolution).
  """

  @behaviour Nx.Backend

  defstruct [:ref]

  # Per-process default stream cache
  @stream_key {__MODULE__, :default_stream}

  @doc false
  def default_stream do
    case Process.get(@stream_key) do
      nil ->
        {:ok, stream} = Mlx.NIF.default_gpu_stream()
        Process.put(@stream_key, stream)
        stream

      stream ->
        stream
    end
  end

  # --- Helpers ---

  defp from_ref(%Nx.Tensor{data: %__MODULE__{ref: ref}}), do: ref
  defp from_ref(%Nx.Tensor{} = t), do: Nx.backend_transfer(t, __MODULE__) |> from_ref()

  defp to_nx(%Nx.Tensor{} = out, ref) do
    %{out | data: %__MODULE__{ref: ref}}
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, reason}), do: raise("MLX error: #{reason}")

  defp s, do: default_stream()

  # Convert a value to a plain integer — Nx 0.10 may pass scalar tensors
  defp to_int(%Nx.Tensor{} = t), do: Nx.to_number(t)
  defp to_int(i) when is_integer(i), do: i

  # CPU stream for linalg ops (many are CPU-only in MLX)
  @cpu_stream_key {__MODULE__, :cpu_stream}
  defp cpu_s do
    case Process.get(@cpu_stream_key) do
      nil ->
        {:ok, stream} = Mlx.NIF.default_cpu_stream()
        Process.put(@cpu_stream_key, stream)
        stream

      stream ->
        stream
    end
  end

  defp mlx_dtype(%Nx.Tensor{type: type}), do: Mlx.Dtype.to_mlx(type)

  # --- Nx.Backend callbacks ---

  @impl true
  def init(opts), do: opts

  @impl true
  def constant(%Nx.Tensor{shape: shape, type: type} = out, number, _backend_options) do
    # Create a scalar, then broadcast to shape
    mlx_type = Mlx.Dtype.to_mlx(type)
    bin = number_to_binary(number, type)
    ref = unwrap!(Mlx.NIF.from_binary(bin, [], mlx_type))

    if shape == {} do
      to_nx(out, ref)
    else
      shape_list = Tuple.to_list(shape)
      result = unwrap!(Mlx.NIF.mlx_broadcast_to(ref, shape_list, s()))
      to_nx(out, result)
    end
  end

  @impl true
  def from_binary(%Nx.Tensor{shape: shape, type: type} = out, binary, _backend_options) do
    mlx_type = Mlx.Dtype.to_mlx(type)
    shape_list = Tuple.to_list(shape)
    ref = unwrap!(Mlx.NIF.from_binary(binary, shape_list, mlx_type))
    to_nx(out, ref)
  end

  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{ref: ref}}, _limit) do
    unwrap!(Mlx.NIF.to_binary(ref))
  end

  @impl true
  def backend_copy(tensor, module, backend_options) do
    if module == __MODULE__ do
      tensor
    else
      binary = to_binary(tensor, 0)
      module.from_binary(tensor, binary, backend_options)
    end
  end

  @impl true
  def backend_transfer(tensor, module, backend_options) do
    result = backend_copy(tensor, module, backend_options)
    backend_deallocate(tensor)
    result
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %__MODULE__{}}), do: :ok

  @impl true
  def inspect(%Nx.Tensor{} = tensor, inspect_opts) do
    limit = if(inspect_opts.limit == :infinity, do: 0, else: inspect_opts.limit + 1)

    tensor
    |> to_binary(limit)
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
  end

  @impl true
  def eye(%Nx.Tensor{shape: {n, m}, type: type} = out, _backend_options) do
    ref = unwrap!(Mlx.NIF.eye(n, m, 0, Mlx.Dtype.to_mlx(type), s()))
    to_nx(out, ref)
  end

  @impl true
  def iota(%Nx.Tensor{shape: shape, type: type} = out, axis, _backend_options) do
    mlx_type = Mlx.Dtype.to_mlx(type)
    shape_list = Tuple.to_list(shape)

    if axis == nil do
      total = Nx.size(shape)
      ref = unwrap!(Mlx.NIF.arange(0.0, total / 1, 1.0, mlx_type, s()))
      result = unwrap!(Mlx.NIF.mlx_reshape(ref, shape_list, s()))
      to_nx(out, result)
    else
      axis_size = elem(shape, axis)
      ref = unwrap!(Mlx.NIF.arange(0.0, axis_size / 1, 1.0, mlx_type, s()))

      broadcast_shape =
        for i <- 0..(tuple_size(shape) - 1) do
          if i == axis, do: axis_size, else: 1
        end

      reshaped = unwrap!(Mlx.NIF.mlx_reshape(ref, broadcast_shape, s()))
      result = unwrap!(Mlx.NIF.mlx_broadcast_to(reshaped, shape_list, s()))
      to_nx(out, result)
    end
  end

  # --- Unary element-wise ops ---

  unary_ops = [
    {:negate, :mlx_negative},
    {:abs, :mlx_abs},
    {:sign, :mlx_sign},
    {:ceil, :mlx_ceil},
    {:floor, :mlx_floor},
    {:round, :mlx_round},
    {:exp, :mlx_exp},
    {:expm1, :mlx_expm1},
    {:log, :mlx_log},
    {:log1p, :mlx_log1p},
    {:sqrt, :mlx_sqrt},
    {:rsqrt, :mlx_rsqrt},
    {:sin, :mlx_sin},
    {:cos, :mlx_cos},
    {:tan, :mlx_tan},
    {:asin, :mlx_arcsin},
    {:acos, :mlx_arccos},
    {:atan, :mlx_arctan},
    {:sinh, :mlx_sinh},
    {:cosh, :mlx_cosh},
    {:tanh, :mlx_tanh},
    {:asinh, :mlx_arcsinh},
    {:acosh, :mlx_arccosh},
    {:atanh, :mlx_arctanh},
    {:erf, :mlx_erf},
    {:erfc, nil},
    {:erf_inv, :mlx_erfinv},
    {:is_nan, :mlx_isnan},
    {:is_infinity, :mlx_isinf},
    {:logical_not, :mlx_logical_not}
  ]

  for {nx_op, mlx_op} <- unary_ops, mlx_op != nil do
    @impl true
    def unquote(nx_op)(out, tensor) do
      ref = from_ref(tensor)
      result = unwrap!(apply(Mlx.NIF, unquote(mlx_op), [ref, s()]))
      to_nx(out, result)
    end
  end

  # sigmoid(x) = 1 / (1 + exp(-x)), using native mlx_sigmoid
  @impl true
  def sigmoid(out, tensor) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_sigmoid(ref, s()))
    to_nx(out, result)
  end

  # erfc = 1 - erf(x), computed via Nx fallback
  @impl true
  def erfc(out, tensor) do
    ref = from_ref(tensor)
    erf_result = unwrap!(Mlx.NIF.mlx_erf(ref, s()))

    one_bin = number_to_binary(1, out.type)
    one = unwrap!(Mlx.NIF.from_binary(one_bin, [], mlx_dtype(out)))

    result = unwrap!(Mlx.NIF.mlx_subtract(one, erf_result, s()))
    to_nx(out, result)
  end

  # --- Binary element-wise ops ---

  binary_ops = [
    {:add, :mlx_add},
    {:subtract, :mlx_subtract},
    {:multiply, :mlx_multiply},
    {:divide, :mlx_divide},
    {:quotient, :mlx_floor_divide},
    {:pow, :mlx_power},
    {:atan2, :mlx_arctan2},
    {:equal, :mlx_equal},
    {:not_equal, :mlx_not_equal},
    {:less, :mlx_less},
    {:less_equal, :mlx_less_equal},
    {:greater, :mlx_greater},
    {:greater_equal, :mlx_greater_equal},
    {:logical_and, :mlx_logical_and},
    {:logical_or, :mlx_logical_or}
  ]

  for {nx_op, mlx_op} <- binary_ops do
    @impl true
    def unquote(nx_op)(out, left, right) do
      l = from_ref(left)
      r = from_ref(right)
      result = unwrap!(apply(Mlx.NIF, unquote(mlx_op), [l, r, s()]))
      to_nx(out, result)
    end
  end

  # Bitwise ops need explicit type casting because MLX's type promotion
  # can promote mixed integer types (e.g. u64 + s32) to float32, which
  # bitwise ops reject. We cast both operands to the output type first.
  @bitwise_ops [
    {:bitwise_and, :mlx_bitwise_and},
    {:bitwise_or, :mlx_bitwise_or},
    {:bitwise_xor, :mlx_bitwise_xor},
    {:left_shift, :mlx_left_shift},
    {:right_shift, :mlx_right_shift}
  ]

  for {nx_op, mlx_op} <- @bitwise_ops do
    @impl true
    def unquote(nx_op)(%Nx.Tensor{type: out_type} = out, left, right) do
      mlx_type = Mlx.Dtype.to_mlx(out_type)
      l = unwrap!(Mlx.NIF.mlx_as_type(from_ref(left), mlx_type, s()))
      r = unwrap!(Mlx.NIF.mlx_as_type(from_ref(right), mlx_type, s()))
      result = unwrap!(apply(Mlx.NIF, unquote(mlx_op), [l, r, s()]))
      to_nx(out, result)
    end
  end

  # remainder: native mlx_remainder
  @impl true
  def remainder(out, left, right) do
    l = from_ref(left)
    r = from_ref(right)
    result = unwrap!(Mlx.NIF.mlx_remainder(l, r, s()))
    to_nx(out, result)
  end

  # --- Aggregation / Reduction ops ---

  @impl true
  def sum(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    reduce_op(:mlx_sum, out, tensor, opts)
  end

  @impl true
  def product(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    reduce_op(:mlx_prod, out, tensor, opts)
  end

  @impl true
  def reduce_max(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    reduce_op(:mlx_max, out, tensor, opts)
  end

  @impl true
  def reduce_min(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    reduce_op(:mlx_min, out, tensor, opts)
  end

  @impl true
  def argmax(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    axis = opts[:axis]
    keepdims = if(opts[:keep_axis], do: true, else: false)
    ref = from_ref(tensor)

    result =
      if axis == nil do
        # No axis specified: flatten and find argmax over all elements
        total = Nx.size(tensor.shape)
        flat = unwrap!(Mlx.NIF.mlx_reshape(ref, [total], s()))
        unwrap!(Mlx.NIF.mlx_argmax(flat, 0, keepdims, s()))
      else
        unwrap!(Mlx.NIF.mlx_argmax(ref, axis, keepdims, s()))
      end

    to_nx(out, result)
  end

  @impl true
  def argmin(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    axis = opts[:axis]
    keepdims = if(opts[:keep_axis], do: true, else: false)
    ref = from_ref(tensor)

    result =
      if axis == nil do
        total = Nx.size(tensor.shape)
        flat = unwrap!(Mlx.NIF.mlx_reshape(ref, [total], s()))
        unwrap!(Mlx.NIF.mlx_argmin(flat, 0, keepdims, s()))
      else
        unwrap!(Mlx.NIF.mlx_argmin(ref, axis, keepdims, s()))
      end

    to_nx(out, result)
  end

  defp reduce_op(nif_fn, out, tensor, opts) do
    axes = opts[:axes]
    keepdims = if(opts[:keep_axes], do: true, else: false)
    ref = from_ref(tensor)
    result = unwrap!(apply(Mlx.NIF, nif_fn, [ref, axes, keepdims, s()]))
    to_nx(out, result)
  end

  # --- Shape ops ---

  @impl true
  def reshape(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    ref = from_ref(tensor)
    shape_list = Tuple.to_list(out.shape)
    result = unwrap!(Mlx.NIF.mlx_reshape(ref, shape_list, s()))
    to_nx(out, result)
  end

  @impl true
  def transpose(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, axes) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_transpose(ref, axes, s()))
    to_nx(out, result)
  end

  @impl true
  def squeeze(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, axes) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_squeeze(ref, axes, s()))
    to_nx(out, result)
  end

  @impl true
  def as_type(%Nx.Tensor{type: type} = out, %Nx.Tensor{} = tensor) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_as_type(ref, Mlx.Dtype.to_mlx(type), s()))
    to_nx(out, result)
  end

  @impl true
  def broadcast(%Nx.Tensor{shape: shape} = out, %Nx.Tensor{} = tensor, _shape, _axes) do
    ref = from_ref(tensor)
    shape_list = Tuple.to_list(shape)
    result = unwrap!(Mlx.NIF.mlx_broadcast_to(ref, shape_list, s()))
    to_nx(out, result)
  end

  @impl true
  def slice(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, start_indices, lengths, strides) do
    ref = from_ref(tensor)

    # Nx 0.10 may pass start_indices as Nx tensors — convert to integers
    starts = Enum.map(start_indices, &to_int/1)

    stop_indices =
      Enum.zip_with(starts, lengths, fn start, len -> start + len end)

    result = unwrap!(Mlx.NIF.mlx_slice(ref, starts, stop_indices, strides, s()))
    to_nx(out, result)
  end

  @impl true
  def concatenate(%Nx.Tensor{} = out, tensors, axis) do
    refs = Enum.map(tensors, &from_ref/1)
    result = unwrap!(Mlx.NIF.mlx_concatenate(refs, axis, s()))
    to_nx(out, result)
  end

  # --- Clip ---

  @impl true
  def clip(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, %Nx.Tensor{} = min, %Nx.Tensor{} = max) do
    ref = from_ref(tensor)
    min_ref = from_ref(min)
    max_ref = from_ref(max)
    result = unwrap!(Mlx.NIF.mlx_clip(ref, min_ref, max_ref, s()))
    to_nx(out, result)
  end

  # --- Sort / Argsort ---

  @impl true
  def sort(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    axis = opts[:axis]
    direction = opts[:direction] || :asc
    ref = from_ref(tensor)
    sorted = unwrap!(Mlx.NIF.mlx_sort(ref, axis, s()))

    result =
      if direction == :desc do
        # Reverse along the sort axis via slicing
        reverse_along_axis(sorted, axis, out.shape)
      else
        sorted
      end

    to_nx(out, result)
  end

  @impl true
  def argsort(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    axis = opts[:axis]
    direction = opts[:direction] || :asc
    ref = from_ref(tensor)
    sorted = unwrap!(Mlx.NIF.mlx_argsort(ref, axis, s()))

    result =
      if direction == :desc do
        reverse_along_axis(sorted, axis, out.shape)
      else
        sorted
      end

    to_nx(out, result)
  end

  # --- Reverse ---
  # mlx_flip is not available in mlx-c v0.1.2, implement via slicing

  @impl true
  def reverse(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, axes) do
    ref = from_ref(tensor)

    result =
      Enum.reduce(axes, ref, fn axis, acc ->
        reverse_along_axis(acc, axis, tensor.shape)
      end)

    to_nx(out, result)
  end

  defp reverse_along_axis(ref, axis, shape) do
    # Negative-stride slicing doesn't work in mlx-c v0.1.2, so we use
    # take with reversed indices instead: [n-1, n-2, ..., 1, 0]
    # mlx_take with axis != 0 also has issues, so we transpose the target
    # axis to front, take on axis 0, then transpose back.
    axis_size = elem(shape, axis)

    indices_bin =
      for i <- (axis_size - 1)..0//-1, into: <<>> do
        <<i::signed-native-32>>
      end

    indices_ref = unwrap!(Mlx.NIF.from_binary(indices_bin, [axis_size], :s32))
    take_on_axis(ref, indices_ref, axis, shape)
  end

  # mlx_take with axis != 0 doesn't work correctly in mlx-c v0.1.2.
  # Workaround: transpose target axis to front, take on axis 0, transpose back.
  defp take_on_axis(ref, indices_ref, 0, _shape) do
    unwrap!(Mlx.NIF.mlx_take(ref, indices_ref, 0, s()))
  end

  defp take_on_axis(ref, indices_ref, axis, shape) do
    ndim = tuple_size(shape)
    # Build permutation: [axis, 0, 1, ..., axis-1, axis+1, ..., ndim-1]
    perm = [axis | Enum.filter(0..(ndim - 1), &(&1 != axis))]
    transposed = unwrap!(Mlx.NIF.mlx_transpose(ref, perm, s()))
    taken = unwrap!(Mlx.NIF.mlx_take(transposed, indices_ref, 0, s()))
    # Build inverse permutation to restore original axis order
    inv_perm = perm |> Enum.with_index() |> Enum.sort_by(&elem(&1, 0)) |> Enum.map(&elem(&1, 1))
    unwrap!(Mlx.NIF.mlx_transpose(taken, inv_perm, s()))
  end

  # --- Take / Gather ---

  @impl true
  def take(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, %Nx.Tensor{} = indices, opts) do
    ref = from_ref(tensor)
    idx = from_ref(indices)
    # Nx.Shared.optional passes [axis: n] keyword list, not plain integer
    axis = if is_list(opts), do: opts[:axis], else: opts
    result = take_on_axis(ref, idx, axis, tensor.shape)
    to_nx(out, result)
  end

  @impl true
  def take_along_axis(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, %Nx.Tensor{} = indices, axis) do
    ref = from_ref(tensor)
    idx = from_ref(indices)
    result = unwrap!(Mlx.NIF.mlx_take_along_axis(ref, idx, axis, s()))
    to_nx(out, result)
  end

  # --- Linear Algebra (Nx.Backend callbacks) ---
  # Many MLX linalg ops are CPU-only (no Metal kernels), so we use cpu_s().

  @impl true
  def triangular_solve(%Nx.Tensor{} = out, %Nx.Tensor{} = a, %Nx.Tensor{} = b, opts) do
    lower = Keyword.get(opts, :lower, true)
    left_side = Keyword.get(opts, :left_side, true)
    transform_a = Keyword.get(opts, :transform_a, :none)

    a_ref = from_ref(a)

    # Handle transform_a: transpose A and flip upper/lower
    {a_ref, upper} =
      case transform_a do
        :none ->
          {a_ref, !lower}

        :transpose ->
          transposed = unwrap!(Mlx.NIF.mlx_transpose(a_ref, [1, 0], cpu_s()))
          {transposed, lower}
      end

    if left_side do
      b_ref = from_ref(b)
      result = unwrap!(Mlx.NIF.mlx_linalg_solve_triangular(a_ref, b_ref, upper, cpu_s()))
      to_nx(out, result)
    else
      # Solve X @ A = B → transpose: A^T @ X^T = B^T
      b_ref = from_ref(b)
      b_ndim = tuple_size(b.shape)
      a_ndim = tuple_size(a.shape)
      b_perm = swap_last_two(b_ndim)
      a_perm = swap_last_two(a_ndim)

      bt = unwrap!(Mlx.NIF.mlx_transpose(b_ref, b_perm, cpu_s()))
      at = unwrap!(Mlx.NIF.mlx_transpose(a_ref, a_perm, cpu_s()))
      xt = unwrap!(Mlx.NIF.mlx_linalg_solve_triangular(at, bt, !upper, cpu_s()))
      result = unwrap!(Mlx.NIF.mlx_transpose(xt, b_perm, cpu_s()))
      to_nx(out, result)
    end
  end

  # Swap last two axes: [0, 1, ..., n-2, n-1] → [0, 1, ..., n-1, n-2]
  defp swap_last_two(ndim) when ndim >= 2 do
    axes = Enum.to_list(0..(ndim - 1))

    List.update_at(axes, ndim - 2, fn _ -> ndim - 1 end)
    |> List.update_at(ndim - 1, fn _ -> ndim - 2 end)
  end

  # Convert MLX pivot indices (uint32 1D) to a permutation matrix (float 2D)
  defp pivots_to_perm_matrix(pivots_ref, p_out, _stream) do
    Mlx.NIF.eval(pivots_ref)
    {:ok, bin} = Mlx.NIF.to_binary(pivots_ref)
    indices = for <<i::native-unsigned-32 <- bin>>, do: i
    n = length(indices)
    mlx_type = Mlx.Dtype.to_mlx(p_out.type)

    # Build permutation matrix: P[i, pivots[i]] = 1.0
    rows =
      Enum.map(indices, fn pivot_col ->
        row = List.duplicate(0.0, n)
        List.replace_at(row, pivot_col, 1.0)
      end)

    flat = List.flatten(rows)
    bin_out = for v <- flat, into: <<>>, do: <<v::native-float-32>>
    unwrap!(Mlx.NIF.from_binary(bin_out, [n, n], mlx_type))
  end

  @impl true
  def qr({%Nx.Tensor{} = q_out, %Nx.Tensor{} = r_out}, %Nx.Tensor{} = tensor, _opts) do
    ref = from_ref(tensor)
    {q_ref, r_ref} = unwrap!(Mlx.NIF.mlx_linalg_qr(ref, cpu_s()))
    {to_nx(q_out, q_ref), to_nx(r_out, r_ref)}
  end

  @impl true
  def cholesky(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_linalg_cholesky(ref, false, cpu_s()))
    to_nx(out, result)
  end

  @impl true
  def lu(
        {%Nx.Tensor{} = p_out, %Nx.Tensor{} = l_out, %Nx.Tensor{} = u_out},
        %Nx.Tensor{} = tensor,
        _opts
      ) do
    ref = from_ref(tensor)
    [pivots_ref, l_ref, u_ref] = unwrap!(Mlx.NIF.mlx_linalg_lu(ref, cpu_s()))
    # MLX returns pivot indices (uint32 1D), Nx expects a permutation matrix (float 2D)
    p_ref = pivots_to_perm_matrix(pivots_ref, p_out, cpu_s())
    {to_nx(p_out, p_ref), to_nx(l_out, l_ref), to_nx(u_out, u_ref)}
  end

  @impl true
  def eigh(
        {%Nx.Tensor{} = eigenvals_out, %Nx.Tensor{} = eigenvecs_out},
        %Nx.Tensor{} = tensor,
        _opts
      ) do
    ref = from_ref(tensor)
    {ev_ref, evec_ref} = unwrap!(Mlx.NIF.mlx_linalg_eigh(ref, :L, cpu_s()))
    {to_nx(eigenvals_out, ev_ref), to_nx(eigenvecs_out, evec_ref)}
  end

  @impl true
  def solve(%Nx.Tensor{} = out, %Nx.Tensor{} = a, %Nx.Tensor{} = b) do
    a_ref = from_ref(a)
    b_ref = from_ref(b)
    result = unwrap!(Mlx.NIF.mlx_linalg_solve(a_ref, b_ref, cpu_s()))
    to_nx(out, result)
  end

  @impl true
  def svd(
        {%Nx.Tensor{} = u_out, %Nx.Tensor{} = s_out, %Nx.Tensor{} = vt_out},
        %Nx.Tensor{} = tensor,
        opts
      ) do
    ref = from_ref(tensor)
    [u_ref, s_ref, vt_ref] = unwrap!(Mlx.NIF.mlx_linalg_svd(ref, cpu_s()))

    # mlx_linalg_svd always returns full SVD (U is m×m, Vt is n×n).
    # When full_matrices?: false, Nx expects the reduced/economy form
    # (U is m×k, Vt is k×n where k = min(m,n)). Slice to match.
    {u_ref, vt_ref} =
      if Keyword.get(opts, :full_matrices?, true) do
        {u_ref, vt_ref}
      else
        u_starts = List.duplicate(0, tuple_size(u_out.shape))
        u_stops = Tuple.to_list(u_out.shape)
        u_strides = List.duplicate(1, tuple_size(u_out.shape))
        u_sliced = unwrap!(Mlx.NIF.mlx_slice(u_ref, u_starts, u_stops, u_strides, s()))

        vt_starts = List.duplicate(0, tuple_size(vt_out.shape))
        vt_stops = Tuple.to_list(vt_out.shape)
        vt_strides = List.duplicate(1, tuple_size(vt_out.shape))
        vt_sliced = unwrap!(Mlx.NIF.mlx_slice(vt_ref, vt_starts, vt_stops, vt_strides, s()))

        {u_sliced, vt_sliced}
      end

    {to_nx(u_out, u_ref), to_nx(s_out, s_ref), to_nx(vt_out, vt_ref)}
  end

  # --- Pad ---

  @impl true
  def pad(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, %Nx.Tensor{} = pad_value, padding_config) do
    ref = from_ref(tensor)
    pad_ref = from_ref(pad_value)

    # padding_config is [{low, high, interior}, ...] per axis
    # MLX pad takes separate axes, low_pads, high_pads lists
    # Interior padding not directly supported by mlx_pad - handle via Nx default if needed
    has_interior = Enum.any?(padding_config, fn {_l, _h, i} -> i > 0 end)

    if has_interior do
      # Fall back to BinaryBackend for interior padding
      Nx.BinaryBackend.pad(
        out,
        Nx.backend_transfer(tensor, Nx.BinaryBackend),
        Nx.backend_transfer(pad_value, Nx.BinaryBackend),
        padding_config
      )
    else
      axes = Enum.to_list(0..(length(padding_config) - 1))
      low_pads = Enum.map(padding_config, fn {l, _h, _i} -> l end)
      high_pads = Enum.map(padding_config, fn {_l, h, _i} -> h end)

      result = unwrap!(Mlx.NIF.mlx_pad(ref, axes, low_pads, high_pads, pad_ref, s()))
      to_nx(out, result)
    end
  end

  # --- Linear algebra ---

  @impl true
  def dot(
        %Nx.Tensor{} = out,
        %Nx.Tensor{} = left,
        [lc],
        [],
        %Nx.Tensor{} = right,
        [rc],
        []
      ) do
    # Simple matmul case: single contraction axis, no batch axes
    l = from_ref(left)
    r = from_ref(right)

    # For 2D tensors with standard contraction, use matmul directly
    l_ndim = tuple_size(left.shape)
    r_ndim = tuple_size(right.shape)

    result =
      if l_ndim == 2 and r_ndim == 2 and lc == 1 and rc == 0 do
        unwrap!(Mlx.NIF.mlx_matmul(l, r, s()))
      else
        # General case: transpose axes then matmul
        # Move contraction axis to last for left, first for right
        l_axes = Enum.to_list(0..(l_ndim - 1)) |> List.delete(lc) |> Kernel.++([lc])
        r_axes = [rc | Enum.to_list(0..(r_ndim - 1)) |> List.delete(rc)]

        lt = unwrap!(Mlx.NIF.mlx_transpose(l, l_axes, s()))
        rt = unwrap!(Mlx.NIF.mlx_transpose(r, r_axes, s()))

        # Reshape to 2D for matmul
        l_rows = div(Nx.size(left.shape), elem(left.shape, lc))
        r_cols = div(Nx.size(right.shape), elem(right.shape, rc))

        lt2 = unwrap!(Mlx.NIF.mlx_reshape(lt, [l_rows, elem(left.shape, lc)], s()))
        rt2 = unwrap!(Mlx.NIF.mlx_reshape(rt, [elem(right.shape, rc), r_cols], s()))

        mm = unwrap!(Mlx.NIF.mlx_matmul(lt2, rt2, s()))
        unwrap!(Mlx.NIF.mlx_reshape(mm, Tuple.to_list(out.shape), s()))
      end

    to_nx(out, result)
  end

  def dot(out, left, contract_left, batch_left, right, contract_right, batch_right) do
    # Fall back to Nx default for batched dot and multi-axis contractions
    Nx.BinaryBackend.dot(
      out,
      Nx.backend_transfer(left, Nx.BinaryBackend),
      contract_left,
      batch_left,
      Nx.backend_transfer(right, Nx.BinaryBackend),
      contract_right,
      batch_right
    )
  end

  # --- Selection ---

  @impl true
  def select(%Nx.Tensor{} = out, pred, on_true, on_false) do
    p = from_ref(pred)
    t = from_ref(on_true)
    f = from_ref(on_false)
    result = unwrap!(Mlx.NIF.mlx_where(p, t, f, s()))
    to_nx(out, result)
  end

  # --- Element-wise max/min (binary) ---

  @impl true
  def max(%Nx.Tensor{} = out, %Nx.Tensor{} = left, %Nx.Tensor{} = right) do
    l = from_ref(left)
    r = from_ref(right)
    result = unwrap!(Mlx.NIF.mlx_maximum(l, r, s()))
    to_nx(out, result)
  end

  @impl true
  def min(%Nx.Tensor{} = out, %Nx.Tensor{} = left, %Nx.Tensor{} = right) do
    l = from_ref(left)
    r = from_ref(right)
    result = unwrap!(Mlx.NIF.mlx_minimum(l, r, s()))
    to_nx(out, result)
  end

  # --- Bitwise not (unary) ---

  @impl true
  def bitwise_not(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_bitwise_invert(ref, s()))
    to_nx(out, result)
  end

  # --- Logical xor (composed via not_equal on boolean) ---

  @impl true
  def logical_xor(%Nx.Tensor{} = out, %Nx.Tensor{} = left, %Nx.Tensor{} = right) do
    l = from_ref(left)
    r = from_ref(right)
    result = unwrap!(Mlx.NIF.mlx_not_equal(l, r, s()))
    to_nx(out, result)
  end

  # --- Boolean reductions: all / any ---

  @impl true
  def all(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    reduce_op(:mlx_all, out, tensor, opts)
  end

  @impl true
  def any(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    reduce_op(:mlx_any, out, tensor, opts)
  end

  # --- Complex number ops ---

  @impl true
  def conjugate(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_conjugate(ref, s()))
    to_nx(out, result)
  end

  @impl true
  def real(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_real(ref, s()))
    to_nx(out, result)
  end

  @impl true
  def imag(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_imag(ref, s()))
    to_nx(out, result)
  end

  # --- Stack ---

  @impl true
  def stack(%Nx.Tensor{} = out, tensors, axis) do
    refs = Enum.map(tensors, &from_ref/1)
    result = unwrap!(Mlx.NIF.mlx_stack(refs, axis, s()))
    to_nx(out, result)
  end

  # --- Put slice ---

  @impl true
  def put_slice(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, start_indices, %Nx.Tensor{} = slice) do
    ref = from_ref(tensor)
    slice_ref = from_ref(slice)

    # Nx 0.10 may pass start_indices as Nx tensors — convert to integers
    starts = Enum.map(start_indices, &to_int/1)

    # Compute stop indices from start + slice shape
    slice_shape = Tuple.to_list(slice.shape)

    stop_indices =
      Enum.zip_with(starts, slice_shape, fn start, len -> start + len end)

    strides = List.duplicate(1, length(starts))

    result =
      unwrap!(Mlx.NIF.mlx_slice_update(ref, slice_ref, starts, stop_indices, strides, s()))

    to_nx(out, result)
  end

  # --- Bitcast ---

  @impl true
  def bitcast(%Nx.Tensor{type: type} = out, %Nx.Tensor{} = tensor) do
    ref = from_ref(tensor)
    result = unwrap!(Mlx.NIF.mlx_view(ref, Mlx.Dtype.to_mlx(type), s()))
    to_nx(out, result)
  end

  # --- Cube root (composed via pow(x, 1/3)) ---

  @impl true
  def cbrt(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    ref = from_ref(tensor)
    # Create scalar 1/3
    third_bin = number_to_binary(1.0 / 3.0, out.type)
    third = unwrap!(Mlx.NIF.from_binary(third_bin, [], mlx_dtype(out)))
    result = unwrap!(Mlx.NIF.mlx_power(ref, third, s()))
    to_nx(out, result)
  end

  # --- FFT / IFFT ---

  @impl true
  def fft(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    ref = from_ref(tensor)
    axis = opts[:axis]
    n = opts[:length] || elem(tensor.shape, axis)
    result = unwrap!(Mlx.NIF.mlx_fft(ref, n, axis, s()))
    to_nx(out, result)
  end

  @impl true
  def ifft(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    ref = from_ref(tensor)
    axis = opts[:axis]
    n = opts[:length] || elem(tensor.shape, axis)
    result = unwrap!(Mlx.NIF.mlx_ifft(ref, n, axis, s()))
    to_nx(out, result)
  end

  # --- Gather ---

  @impl true
  def gather(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, %Nx.Tensor{} = indices, opts) do
    ref = from_ref(tensor)
    idx_ref = from_ref(indices)
    axes = opts[:axes]
    k = length(axes)

    idx_arrays = split_indices_along_last_axis(idx_ref, indices.shape, k)

    # Build slice_sizes: 1 for gathered axes, full size for others
    input_shape = Tuple.to_list(tensor.shape)

    slice_sizes =
      for {dim, i} <- Enum.with_index(input_shape) do
        if i in axes, do: 1, else: dim
      end

    result = unwrap!(Mlx.NIF.mlx_gather(ref, idx_arrays, axes, slice_sizes, s()))

    # Squeeze gathered dimensions from output
    # MLX output shape is (*idx_batch_shape, *slice_sizes)
    idx_ndim = tuple_size(indices.shape) - 1
    squeeze_axes = Enum.map(axes, fn ax -> idx_ndim + ax end)
    result = unwrap!(Mlx.NIF.mlx_squeeze(result, Enum.sort(squeeze_axes), s()))

    to_nx(out, result)
  end

  # --- Indexed add / put ---

  @impl true
  def indexed_add(
        %Nx.Tensor{} = out,
        %Nx.Tensor{} = tensor,
        %Nx.Tensor{} = indices,
        %Nx.Tensor{} = updates,
        opts
      ) do
    scatter_op(:mlx_scatter_add, out, tensor, indices, updates, opts)
  end

  @impl true
  def indexed_put(
        %Nx.Tensor{} = out,
        %Nx.Tensor{} = tensor,
        %Nx.Tensor{} = indices,
        %Nx.Tensor{} = updates,
        opts
      ) do
    scatter_op(:mlx_scatter, out, tensor, indices, updates, opts)
  end

  defp scatter_op(nif_fn, out, tensor, indices, updates, opts) do
    ref = from_ref(tensor)
    idx_ref = from_ref(indices)
    upd_ref = from_ref(updates)
    axes = opts[:axes]

    k = length(axes)
    idx_arrays = split_indices_along_last_axis(idx_ref, indices.shape, k)

    # Reshape updates: Nx gives (*batch, *remaining), MLX wants (*batch, *slice_sizes)
    upd_ref = reshape_updates_for_scatter(upd_ref, indices.shape, tensor.shape, axes)

    result = unwrap!(apply(Mlx.NIF, nif_fn, [ref, idx_arrays, upd_ref, axes, s()]))
    to_nx(out, result)
  end

  defp split_indices_along_last_axis(idx_ref, idx_shape, k) do
    ndim = tuple_size(idx_shape)
    batch_dims = Tuple.to_list(idx_shape) |> Enum.take(ndim - 1)

    for i <- 0..(k - 1) do
      start = List.duplicate(0, ndim - 1) ++ [i]
      stop = batch_dims ++ [i + 1]
      strides = List.duplicate(1, ndim)
      sliced = unwrap!(Mlx.NIF.mlx_slice(idx_ref, start, stop, strides, s()))
      unwrap!(Mlx.NIF.mlx_squeeze(sliced, [ndim - 1], s()))
    end
  end

  defp reshape_updates_for_scatter(upd_ref, indices_shape, input_shape, axes) do
    idx_ndim = tuple_size(indices_shape) - 1
    idx_batch = Tuple.to_list(indices_shape) |> Enum.take(idx_ndim)

    slice_sizes =
      input_shape
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.map(fn {dim, i} -> if i in axes, do: 1, else: dim end)

    target_shape = idx_batch ++ slice_sizes
    unwrap!(Mlx.NIF.mlx_reshape(upd_ref, target_shape, s()))
  end

  # --- Convolution ---
  # Satisfies: B1 (Nx.Backend callbacks)

  @impl true
  def conv(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, %Nx.Tensor{} = kernel, opts) do
    batch_group_size = opts[:batch_group_size] || 1

    if batch_group_size > 1 do
      # batch_group_size > 1 not supported by MLX, fall back to BinaryBackend
      Nx.BinaryBackend.conv(
        out,
        Nx.backend_transfer(tensor, Nx.BinaryBackend),
        Nx.backend_transfer(kernel, Nx.BinaryBackend),
        opts
      )
    else
      do_conv(out, tensor, kernel, opts)
    end
  end

  defp do_conv(out, tensor, kernel, opts) do
    input_ref = from_ref(tensor)
    kernel_ref = from_ref(kernel)

    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    groups = opts[:feature_group_size] || 1
    input_perm = opts[:input_permutation]
    kernel_perm = opts[:kernel_permutation]
    output_perm = opts[:output_permutation]

    ndim = tuple_size(tensor.shape)

    # Step 1: Apply input/kernel permutations to get standard NCHW/OIHW format
    input_std = unwrap!(Mlx.NIF.mlx_transpose(input_ref, input_perm, s()))
    kernel_std = unwrap!(Mlx.NIF.mlx_transpose(kernel_ref, kernel_perm, s()))

    # Step 2: Convert NCHW to channels-last (N, spatial..., C) for MLX
    nchw_to_cl = [0] ++ Enum.to_list(2..(ndim - 1)) ++ [1]
    input_cl = unwrap!(Mlx.NIF.mlx_transpose(input_std, nchw_to_cl, s()))
    kernel_cl = unwrap!(Mlx.NIF.mlx_transpose(kernel_std, nchw_to_cl, s()))

    # Step 3: Split padding into low and high
    pad_lo = Enum.map(padding, fn {lo, _} -> lo end)
    pad_hi = Enum.map(padding, fn {_, hi} -> hi end)

    # Step 4: Call MLX conv_general (flip=false for cross-correlation)
    result =
      unwrap!(
        Mlx.NIF.mlx_conv_general(
          input_cl,
          kernel_cl,
          strides,
          pad_lo,
          pad_hi,
          kernel_dilation,
          input_dilation,
          groups,
          false,
          s()
        )
      )

    # Step 5: Convert output from channels-last back to NCHW
    cl_to_nchw = [0, ndim - 1] ++ Enum.to_list(1..(ndim - 2))
    result_nchw = unwrap!(Mlx.NIF.mlx_transpose(result, cl_to_nchw, s()))

    # Step 6: Apply output permutation
    result_final = unwrap!(Mlx.NIF.mlx_transpose(result_nchw, output_perm, s()))

    to_nx(out, result_final)
  end

  # --- To Batched ---

  @impl true
  def to_batched(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    ref = from_ref(tensor)
    leftover = opts[:leftover] || :repeat

    batch_size = elem(out.shape, 0)
    total = elem(tensor.shape, 0)

    if total == 0 do
      []
    else
      {split_ref, num_splits} =
        case leftover do
          :discard ->
            n = div(total, batch_size)
            keep = n * batch_size

            trimmed =
              if keep < total do
                start = List.duplicate(0, tuple_size(tensor.shape))
                stop = [keep | tl(Tuple.to_list(tensor.shape))]
                strides = List.duplicate(1, tuple_size(tensor.shape))
                unwrap!(Mlx.NIF.mlx_slice(ref, start, stop, strides, s()))
              else
                ref
              end

            {trimmed, n}

          :repeat ->
            if rem(total, batch_size) == 0 do
              {ref, div(total, batch_size)}
            else
              n = div(total + batch_size - 1, batch_size)
              pad_amount = n * batch_size - total

              start = List.duplicate(0, tuple_size(tensor.shape))
              stop = [pad_amount | tl(Tuple.to_list(tensor.shape))]
              strides = List.duplicate(1, tuple_size(tensor.shape))
              pad_slice = unwrap!(Mlx.NIF.mlx_slice(ref, start, stop, strides, s()))
              padded = unwrap!(Mlx.NIF.mlx_concatenate([ref, pad_slice], 0, s()))
              {padded, n}
            end
        end

      if num_splits == 0 do
        []
      else
        parts = unwrap!(Mlx.NIF.mlx_split_equal_parts(split_ref, num_splits, 0, s()))
        Enum.map(parts, fn part_ref -> to_nx(out, part_ref) end)
      end
    end
  end

  # --- Private helpers ---

  defp number_to_binary(number, type) when number in [:infinity, :neg_infinity, :nan] do
    # Handle non-finite float constants (infinity, neg_infinity, nan)
    case type do
      {:f, size} -> Nx.Shared.write_non_finite(number, size)
      {:bf, 16} -> Nx.Shared.write_non_finite_bf16(number)
      {:c, 64} -> Nx.Shared.write_non_finite(number, 32) <> <<0.0::float-native-32>>
      _ -> number_to_binary(0, type)
    end
  end

  defp number_to_binary(number, {type, size}) do
    # Encode a single number as binary in the given Nx type
    case {type, size} do
      {:u, 8} ->
        <<number::unsigned-native-8>>

      {:u, 16} ->
        <<number::unsigned-native-16>>

      {:u, 32} ->
        <<number::unsigned-native-32>>

      {:u, 64} ->
        <<number::unsigned-native-64>>

      {:s, 8} ->
        <<number::signed-native-8>>

      {:s, 16} ->
        <<number::signed-native-16>>

      {:s, 32} ->
        <<number::signed-native-32>>

      {:s, 64} ->
        <<number::signed-native-64>>

      {:f, 16} ->
        <<number::float-native-16>>

      {:f, 32} ->
        <<number::float-native-32>>

      {:bf, 16} ->
        # bf16: truncate f32 to upper 16 bits
        <<_::16, upper::binary-size(2)>> = <<number::float-native-32>>
        upper

      {:c, 64} ->
        real = if is_number(number), do: number / 1, else: 0.0
        <<real::float-native-32, 0.0::float-native-32>>

      {:pred, 8} ->
        <<if(number != 0, do: 1, else: 0)::native-8>>
    end
  end

  # --- Cumulative ops ---

  @impl true
  def cumulative_sum(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    cumulative_op(:mlx_cumsum, out, tensor, opts)
  end

  @impl true
  def cumulative_product(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    cumulative_op(:mlx_cumprod, out, tensor, opts)
  end

  @impl true
  def cumulative_max(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    cumulative_op(:mlx_cummax, out, tensor, opts)
  end

  @impl true
  def cumulative_min(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    cumulative_op(:mlx_cummin, out, tensor, opts)
  end

  defp cumulative_op(nif_fn, out, tensor, opts) do
    axis = opts[:axis]
    reverse = if(opts[:reverse], do: true, else: false)
    # Nx cumulative ops are always inclusive
    inclusive = true
    ref = from_ref(tensor)
    result = unwrap!(apply(Mlx.NIF, nif_fn, [ref, axis, reverse, inclusive, s()]))
    to_nx(out, result)
  end

  # --- Not yet implemented callbacks ---
  # These raise clear errors instead of generating compiler warnings.

  # --- Bit operations (via BinaryBackend) ---
  # No mlx-c equivalent; transfer data to BinaryBackend for computation.

  @impl true
  def count_leading_zeros(out, tensor) do
    with_binary_backend(fn ->
      Nx.BinaryBackend.count_leading_zeros(
        out,
        Nx.backend_transfer(tensor, Nx.BinaryBackend)
      )
    end)
  end

  @impl true
  def population_count(out, tensor) do
    with_binary_backend(fn ->
      Nx.BinaryBackend.population_count(
        out,
        Nx.backend_transfer(tensor, Nx.BinaryBackend)
      )
    end)
  end

  # --- General reduce (via BinaryBackend) ---
  # Requires user-defined functions; mlx-c can't accept Elixir closures.

  @impl true
  def reduce(out, tensor, acc, opts, fun) do
    with_binary_backend(fn ->
      Nx.BinaryBackend.reduce(
        out,
        Nx.backend_transfer(tensor, Nx.BinaryBackend),
        Nx.backend_transfer(acc, Nx.BinaryBackend),
        opts,
        fun
      )
    end)
  end

  # --- Pointer operations (not supported) ---
  # MLX arrays are managed by the runtime; raw pointer access is not meaningful.
  # Even Nx.BinaryBackend raises for these. Use Nx.from_binary/3 or Nx.to_binary/1 instead.

  @impl true
  def from_pointer(_arg1, _arg2, _arg3, _arg4, _arg5) do
    raise ArgumentError,
          "from_pointer is not supported by Mlx.Backend. " <>
            "MLX arrays are managed by the MLX runtime. Use Nx.from_binary/3 instead."
  end

  @impl true
  def to_pointer(_tensor, _opts) do
    raise ArgumentError,
          "to_pointer is not supported by Mlx.Backend. " <>
            "MLX arrays are managed by the MLX runtime. Use Nx.to_binary/1 instead."
  end

  # Window ops — implemented via as_strided + reductions

  @impl true
  def window_sum(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, window_dimensions, opts) do
    window_reduce_impl(out, tensor, window_dimensions, opts, :sum)
  end

  @impl true
  def window_max(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, window_dimensions, opts) do
    window_reduce_impl(out, tensor, window_dimensions, opts, :max)
  end

  @impl true
  def window_min(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, window_dimensions, opts) do
    window_reduce_impl(out, tensor, window_dimensions, opts, :min)
  end

  @impl true
  def window_product(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, window_dimensions, opts) do
    window_reduce_impl(out, tensor, window_dimensions, opts, :product)
  end

  defp window_reduce_impl(out, tensor, window_dimensions, opts, op) do
    padding = opts[:padding]
    strides = opts[:strides]
    window_dilations = opts[:window_dilations]
    type = tensor.type

    ref = from_ref(tensor)
    ndim = tuple_size(tensor.shape)
    win_dims = Tuple.to_list(window_dimensions)

    # Use CPU stream: GPU reduction on non-contiguous as_strided views
    # produces wrong results (mlx-c v0.1.2 contiguity bug)
    stream = cpu_s()

    # Step 1: Pad the input with appropriate identity value
    pad_ref = window_pad_ref(op, type)
    needs_padding = Enum.any?(padding, fn {lo, hi} -> lo > 0 or hi > 0 end)

    padded_ref =
      if needs_padding do
        axes = Enum.to_list(0..(ndim - 1))
        low_pads = Enum.map(padding, fn {lo, _hi} -> lo end)
        high_pads = Enum.map(padding, fn {_lo, hi} -> hi end)
        unwrap!(Mlx.NIF.mlx_pad(ref, axes, low_pads, high_pads, pad_ref, stream))
      else
        ref
      end

    # Compute padded shape
    padded_shape =
      for i <- 0..(ndim - 1) do
        {lo, hi} = Enum.at(padding, i)
        elem(tensor.shape, i) + lo + hi
      end

    # Step 2: Compute as_strided shape and strides
    elem_strides = compute_elem_strides(padded_shape)

    # Effective window size accounting for dilation
    effective_win =
      Enum.zip(win_dims, window_dilations)
      |> Enum.map(fn {w, d} -> (w - 1) * d + 1 end)

    # Output dimensions
    out_dims =
      for i <- 0..(ndim - 1) do
        div(Enum.at(padded_shape, i) - Enum.at(effective_win, i), Enum.at(strides, i)) + 1
      end

    # as_strided shape: [out_dims..., win_dims...]
    strided_shape = out_dims ++ win_dims

    # as_strided strides: [elem_stride*pool_stride..., elem_stride*dilation...]
    out_strides =
      for i <- 0..(ndim - 1) do
        Enum.at(elem_strides, i) * Enum.at(strides, i)
      end

    win_strides =
      for i <- 0..(ndim - 1) do
        Enum.at(elem_strides, i) * Enum.at(window_dilations, i)
      end

    strided_strides = out_strides ++ win_strides

    strided_ref =
      unwrap!(Mlx.NIF.mlx_as_strided(padded_ref, strided_shape, strided_strides, 0, stream))

    # Step 3: Reduce over window dimensions (last ndim axes)
    reduce_axes = Enum.to_list(ndim..(2 * ndim - 1))

    reduced_ref =
      case op do
        :sum -> unwrap!(Mlx.NIF.mlx_sum(strided_ref, reduce_axes, false, stream))
        :max -> unwrap!(Mlx.NIF.mlx_max(strided_ref, reduce_axes, false, stream))
        :min -> unwrap!(Mlx.NIF.mlx_min(strided_ref, reduce_axes, false, stream))
        :product -> unwrap!(Mlx.NIF.mlx_prod(strided_ref, reduce_axes, false, stream))
      end

    to_nx(out, reduced_ref)
  end

  defp compute_elem_strides(shape) do
    # Row-major (C-contiguous) strides in elements, computed from right
    shape
    |> Enum.reverse()
    |> Enum.reduce({[], 1}, fn dim, {strides, acc} ->
      {[acc | strides], acc * dim}
    end)
    |> elem(0)
  end

  defp window_pad_ref(:sum, type) do
    from_ref(Nx.tensor(0, type: type))
  end

  defp window_pad_ref(:product, type) do
    from_ref(Nx.tensor(1, type: type))
  end

  defp window_pad_ref(:max, type) do
    # Use type min binary for negative infinity / min integer value
    binary = Nx.Type.min_binary(type)
    unwrap!(Mlx.NIF.from_binary(binary, [], Mlx.Dtype.to_mlx(type)))
  end

  defp window_pad_ref(:min, type) do
    # Use type max binary for positive infinity / max integer value
    binary = Nx.Type.max_binary(type)
    unwrap!(Mlx.NIF.from_binary(binary, [], Mlx.Dtype.to_mlx(type)))
  end

  # --- Window ops with user-defined functions (via BinaryBackend) ---
  # mlx-c can't accept Elixir closures; transfer data to BinaryBackend.

  @impl true
  def window_reduce(out, tensor, acc, window_dimensions, opts, fun) do
    with_binary_backend(fn ->
      Nx.BinaryBackend.window_reduce(
        out,
        Nx.backend_transfer(tensor, Nx.BinaryBackend),
        Nx.backend_transfer(acc, Nx.BinaryBackend),
        window_dimensions,
        opts,
        fun
      )
    end)
  end

  @impl true
  def window_scatter_max(out, tensor, source, init_value, window_dimensions, opts) do
    with_binary_backend(fn ->
      Nx.BinaryBackend.window_scatter_max(
        out,
        Nx.backend_transfer(tensor, Nx.BinaryBackend),
        Nx.backend_transfer(source, Nx.BinaryBackend),
        Nx.backend_transfer(init_value, Nx.BinaryBackend),
        window_dimensions,
        opts
      )
    end)
  end

  @impl true
  def window_scatter_min(out, tensor, source, init_value, window_dimensions, opts) do
    with_binary_backend(fn ->
      Nx.BinaryBackend.window_scatter_min(
        out,
        Nx.backend_transfer(tensor, Nx.BinaryBackend),
        Nx.backend_transfer(source, Nx.BinaryBackend),
        Nx.backend_transfer(init_value, Nx.BinaryBackend),
        window_dimensions,
        opts
      )
    end)
  end

  # Temporarily sets the default backend to BinaryBackend so that
  # intermediate tensors created during BinaryBackend operations
  # (e.g., inside user closures or internal Nx dispatch) stay on BinaryBackend.
  defp with_binary_backend(fun) do
    prev = Nx.default_backend()
    Nx.default_backend({Nx.BinaryBackend, []})

    try do
      fun.()
    after
      Nx.default_backend(prev)
    end
  end
end
