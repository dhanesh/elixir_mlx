defmodule Mlx.NIF do
  @moduledoc false
  # Satisfies: T1 (NIF via mlx-c), RT-1 (array lifecycle)

  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:elixir_mlx), ~c"mlx_nif")
    :erlang.load_nif(path, 0)
  end

  # --- Array creation ---
  def from_binary(_data, _shape, _dtype), do: :erlang.nif_error(:not_loaded)
  def zeros(_shape, _dtype, _stream), do: :erlang.nif_error(:not_loaded)
  def ones(_shape, _dtype, _stream), do: :erlang.nif_error(:not_loaded)
  def full(_shape, _val, _dtype, _stream), do: :erlang.nif_error(:not_loaded)
  def eye(_n, _m, _k, _dtype, _stream), do: :erlang.nif_error(:not_loaded)
  def arange(_start, _stop, _step, _dtype, _stream), do: :erlang.nif_error(:not_loaded)
  def linspace(_start, _stop, _num, _dtype, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Array properties ---
  def shape(_arr), do: :erlang.nif_error(:not_loaded)
  def dtype(_arr), do: :erlang.nif_error(:not_loaded)
  def ndim(_arr), do: :erlang.nif_error(:not_loaded)
  def size(_arr), do: :erlang.nif_error(:not_loaded)
  def nbytes(_arr), do: :erlang.nif_error(:not_loaded)

  # --- Array data ---
  def eval(_arr), do: :erlang.nif_error(:not_loaded)
  def to_binary(_arr), do: :erlang.nif_error(:not_loaded)

  # --- Unary ops ---
  def mlx_negative(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_abs(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_sign(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_ceil(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_floor(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_round(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_exp(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_expm1(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_log(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_log2(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_log10(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_log1p(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_sqrt(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_reciprocal(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_sin(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_cos(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_tan(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_arcsin(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_arccos(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_arctan(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_sinh(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_cosh(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_tanh(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_arcsinh(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_arccosh(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_arctanh(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_erf(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_erfinv(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_logical_not(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_isnan(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_isinf(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_sigmoid(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_bitwise_invert(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_conjugate(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_real(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_imag(_arr, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Wave 1: New unary ops ---
  def mlx_rsqrt(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_square(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_degrees(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_radians(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_isfinite(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_isneginf(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_isposinf(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_copy(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_stop_gradient(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_ones_like(_arr, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_zeros_like(_arr, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Binary ops ---
  def mlx_add(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_subtract(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_multiply(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_divide(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_floor_divide(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_power(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_logaddexp(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_arctan2(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_equal(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_not_equal(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_less(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_less_equal(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_greater(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_greater_equal(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_logical_and(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_logical_or(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_matmul(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_maximum(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_minimum(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Wave 1: New binary ops ---
  def mlx_remainder(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_outer(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_inner(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_kron(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Bitwise ops ---
  def mlx_bitwise_and(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_bitwise_or(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_bitwise_xor(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_left_shift(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_right_shift(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Reduction ops ---
  def mlx_sum(_arr, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_prod(_arr, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_mean(_arr, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_min(_arr, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_max(_arr, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_argmax(_arr, _axis, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_argmin(_arr, _axis, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_all(_arr, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_any(_arr, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Wave 1: New reduction/cumulative ops ---
  def mlx_logsumexp(_arr, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_std(_arr, _axes, _keepdims, _ddof, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_var(_arr, _axes, _keepdims, _ddof, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_cumsum(_arr, _axis, _reverse, _inclusive, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_cumprod(_arr, _axis, _reverse, _inclusive, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_cummax(_arr, _axis, _reverse, _inclusive, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_cummin(_arr, _axis, _reverse, _inclusive, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Shape ops ---
  def mlx_reshape(_arr, _shape, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_transpose(_arr, _axes, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_squeeze(_arr, _axes, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_expand_dims(_arr, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_broadcast_to(_arr, _shape, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_flatten(_arr, _start_axis, _end_axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_slice(_arr, _start, _stop, _strides, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Type ops ---
  def mlx_as_type(_arr, _dtype, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Clip, Sort, Take ops ---
  def mlx_clip(_arr, _min, _max, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_sort(_arr, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_argsort(_arr, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_take(_arr, _indices, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_take_along_axis(_arr, _indices, _axis, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Triangular ops ---
  def mlx_triu(_arr, _k, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_tril(_arr, _k, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Pad, Repeat, Tile ops ---
  def mlx_pad(_arr, _axes, _low_pad, _high_pad, _pad_value, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_repeat(_arr, _repeats, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_tile(_arr, _reps, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Concatenate / Stack (uses vector_array) ---
  def mlx_concatenate(_arrays, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_stack(_arrays, _axis, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Slice update ---
  def mlx_slice_update(_src, _update, _start, _stop, _strides, _stream),
    do: :erlang.nif_error(:not_loaded)

  # --- View (bitcast) ---
  def mlx_view(_arr, _dtype, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Selection ops ---
  def mlx_where(_cond, _x, _y, _stream), do: :erlang.nif_error(:not_loaded)

  # --- FFT ops ---
  def mlx_fft(_arr, _n, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_ifft(_arr, _n, _axis, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Gather / Scatter ops ---
  def mlx_gather(_arr, _indices, _axes, _slice_sizes, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_scatter(_arr, _indices, _updates, _axes, _stream), do: :erlang.nif_error(:not_loaded)

  def mlx_scatter_add(_arr, _indices, _updates, _axes, _stream),
    do: :erlang.nif_error(:not_loaded)

  # --- Wave 1: New scatter variants ---
  def mlx_scatter_max(_arr, _indices, _updates, _axes, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_scatter_min(_arr, _indices, _updates, _axes, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_scatter_prod(_arr, _indices, _updates, _axes, _stream),
    do: :erlang.nif_error(:not_loaded)

  # --- Wave 1: Shape/Selection/Comparison/Matrix ops ---
  def mlx_moveaxis(_arr, _src, _dst, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_swapaxes(_arr, _ax1, _ax2, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_diagonal(_arr, _offset, _ax1, _ax2, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_diag(_arr, _k, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_roll(_arr, _shift, _axes, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_unflatten(_arr, _axis, _shape, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_as_strided(_arr, _shape, _strides, _offset, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_identity(_n, _dtype, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_tri(_n, _m, _k, _dtype, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_topk(_arr, _k, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_partition(_arr, _kth, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_argpartition(_arr, _kth, _axis, _stream), do: :erlang.nif_error(:not_loaded)

  def mlx_put_along_axis(_arr, _indices, _values, _axis, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_allclose(_a, _b, _rtol, _atol, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_isclose(_a, _b, _rtol, _atol, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_array_equal(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_nan_to_num(_arr, _nan, _posinf, _neginf, _stream), do: :erlang.nif_error(:not_loaded)

  def mlx_number_of_elements(_arr, _axes, _inverted, _dtype, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_softmax(_arr, _axes, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_tensordot(_a, _b, _axes_a, _axes_b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_addmm(_c, _a, _b, _alpha, _beta, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_trace(_arr, _offset, _ax1, _ax2, _dtype, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_einsum(_subscripts, _operands, _stream), do: :erlang.nif_error(:not_loaded)

  def mlx_block_masked_mm(_a, _b, _block_size, _mask_out, _mask_lhs, _mask_rhs, _stream),
    do: :erlang.nif_error(:not_loaded)

  # --- Convolution ---
  def mlx_conv_general(
        _input,
        _weight,
        _strides,
        _pad_lo,
        _pad_hi,
        _kdil,
        _idil,
        _groups,
        _flip,
        _stream
      ),
      do: :erlang.nif_error(:not_loaded)

  # --- Split (for to_batched) ---
  def mlx_split_equal_parts(_arr, _num_splits, _axis, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Wave 2: Random ops ---
  def mlx_random_key(_seed), do: :erlang.nif_error(:not_loaded)
  def mlx_random_seed(_seed), do: :erlang.nif_error(:not_loaded)
  def mlx_random_split(_key, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_random_split_num(_key, _num, _stream), do: :erlang.nif_error(:not_loaded)

  def mlx_random_uniform(_low, _high, _shape, _dtype, _key_or_nil, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_random_normal(_shape, _dtype, _loc, _scale, _key_or_nil, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_random_bernoulli(_p, _shape, _key_or_nil, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_random_randint(_low, _high, _shape, _dtype, _key_or_nil, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_random_truncated_normal(_lower, _upper, _shape, _dtype, _key_or_nil, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_random_categorical(_logits, _axis, _key_or_nil, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_random_gumbel(_shape, _dtype, _key_or_nil, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_random_laplace(_shape, _dtype, _loc, _scale, _key_or_nil, _stream),
    do: :erlang.nif_error(:not_loaded)

  # --- Wave 3: Linear Algebra ops ---
  def mlx_linalg_inv(_a, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_pinv(_a, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_cholesky(_a, _upper, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_cholesky_inv(_a, _upper, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_qr(_a, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_svd(_a, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_eigh(_a, _uplo, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_eigvalsh(_a, _uplo, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_lu(_a, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_lu_factor(_a, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_solve(_a, _b, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_solve_triangular(_a, _b, _upper, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_cross(_a, _b, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_tri_inv(_a, _upper, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_norm(_a, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_linalg_norm_p(_a, _ord, _axes, _keepdims, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Wave 4: FFT expansion ---
  def mlx_rfft(_arr, _n, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_irfft(_arr, _n, _axis, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_fft2(_arr, _n_list, _axes_list, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_fftn(_arr, _n_list, _axes_list, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_ifft2(_arr, _n_list, _axes_list, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_ifftn(_arr, _n_list, _axes_list, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_rfft2(_arr, _n_list, _axes_list, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_rfftn(_arr, _n_list, _axes_list, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_irfft2(_arr, _n_list, _axes_list, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_irfftn(_arr, _n_list, _axes_list, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Wave 5: Quantization ---
  def mlx_quantize(_w, _group_size, _bits, _stream), do: :erlang.nif_error(:not_loaded)

  def mlx_dequantize(_w, _scales, _biases, _group_size, _bits, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_quantized_matmul(_x, _w, _scales, _biases, _transpose, _group_size, _bits, _stream),
    do: :erlang.nif_error(:not_loaded)

  # --- Wave 6: Fast ops ---
  def mlx_fast_layer_norm(_x, _weight, _bias, _eps, _stream), do: :erlang.nif_error(:not_loaded)
  def mlx_fast_rms_norm(_x, _weight, _eps, _stream), do: :erlang.nif_error(:not_loaded)

  def mlx_fast_rope(_x, _dims, _traditional, _base, _scale, _offset, _freqs, _stream),
    do: :erlang.nif_error(:not_loaded)

  def mlx_fast_sdpa(_q, _k, _v, _scale, _mask, _stream), do: :erlang.nif_error(:not_loaded)

  # --- I/O ---
  def mlx_save(_path, _arr), do: :erlang.nif_error(:not_loaded)
  def mlx_load(_path, _stream), do: :erlang.nif_error(:not_loaded)

  def mlx_save_safetensors(_path, _keys, _arrays, _meta_keys, _meta_vals),
    do: :erlang.nif_error(:not_loaded)

  def mlx_load_safetensors(_path, _stream), do: :erlang.nif_error(:not_loaded)

  # --- Device / Stream ---
  def default_cpu_stream, do: :erlang.nif_error(:not_loaded)
  def default_gpu_stream, do: :erlang.nif_error(:not_loaded)
  def default_device, do: :erlang.nif_error(:not_loaded)
  def set_default_device(_device), do: :erlang.nif_error(:not_loaded)
  def device_new(_type), do: :erlang.nif_error(:not_loaded)
  def synchronize(_stream), do: :erlang.nif_error(:not_loaded)

  # --- Wave 9: Function transforms (closure bridge) ---
  def closure_respond(_bridge_ref, _array_ref_list), do: :erlang.nif_error(:not_loaded)
  def closure_respond_error(_bridge_ref), do: :erlang.nif_error(:not_loaded)
  def value_and_grad_apply(_helper_pid, _inputs, _argnums), do: :erlang.nif_error(:not_loaded)
  def vjp_apply(_helper_pid, _primals, _cotangents), do: :erlang.nif_error(:not_loaded)
  def jvp_apply(_helper_pid, _primals, _tangents), do: :erlang.nif_error(:not_loaded)
  def vmap_apply(_helper_pid, _inputs, _in_axes, _out_axes), do: :erlang.nif_error(:not_loaded)
end
