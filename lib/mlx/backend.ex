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
    {:rsqrt, nil},
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

  # rsqrt = reciprocal(sqrt(x))
  @impl true
  def rsqrt(out, tensor) do
    ref = from_ref(tensor)
    sq = unwrap!(Mlx.NIF.mlx_sqrt(ref, s()))
    result = unwrap!(Mlx.NIF.mlx_reciprocal(sq, s()))
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
    {:logical_or, :mlx_logical_or},
    {:bitwise_and, :mlx_bitwise_and},
    {:bitwise_or, :mlx_bitwise_or},
    {:bitwise_xor, :mlx_bitwise_xor},
    {:left_shift, :mlx_left_shift},
    {:right_shift, :mlx_right_shift}
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

  # remainder: a - floor_divide(a, b) * b
  @impl true
  def remainder(out, left, right) do
    l = from_ref(left)
    r = from_ref(right)
    fd = unwrap!(Mlx.NIF.mlx_floor_divide(l, r, s()))
    prod = unwrap!(Mlx.NIF.mlx_multiply(fd, r, s()))
    result = unwrap!(Mlx.NIF.mlx_subtract(l, prod, s()))
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

    stop_indices =
      Enum.zip_with(start_indices, lengths, fn start, len -> start + len end)

    result = unwrap!(Mlx.NIF.mlx_slice(ref, start_indices, stop_indices, strides, s()))
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

  # --- Triangular ---

  @impl true
  def triangular_solve(_out, _a, _b, _opts) do
    raise "triangular_solve not yet implemented in Mlx.Backend"
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

  # --- Private helpers ---

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

  # --- Not yet implemented callbacks ---
  # These raise clear errors instead of generating compiler warnings.

  @not_implemented_ops ~w(
    all any bitcast bitwise_not cbrt conjugate conv count_leading_zeros
    fft from_pointer gather ifft imag indexed_add indexed_put logical_xor
    max min population_count put_slice real reduce stack to_batched
    to_pointer window_max window_min window_product window_reduce
    window_scatter_max window_scatter_min window_sum
  )a

  for op <- @not_implemented_ops do
    arity =
      case op do
        op when op in [:bitcast, :bitwise_not, :cbrt, :conjugate, :imag, :real, :to_pointer] ->
          2

        op when op in [:all, :any, :fft, :ifft, :logical_xor, :max, :min, :stack, :to_batched] ->
          3

        op
        when op in [
               :conv,
               :gather,
               :put_slice,
               :window_max,
               :window_min,
               :window_product,
               :window_sum
             ] ->
          4

        op when op in [:indexed_add, :indexed_put] ->
          5

        op when op in [:count_leading_zeros, :population_count] ->
          2

        :from_pointer ->
          5

        :reduce ->
          5

        :window_reduce ->
          6

        :window_scatter_max ->
          6

        :window_scatter_min ->
          6

        _ ->
          3
      end

    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(op)(unquote_splicing(args)) do
      raise ArgumentError, "#{unquote(op)} is not yet implemented in Mlx.Backend"
    end
  end
end
