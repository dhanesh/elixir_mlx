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

  # --- Device / Stream ---
  def default_cpu_stream, do: :erlang.nif_error(:not_loaded)
  def default_gpu_stream, do: :erlang.nif_error(:not_loaded)
  def default_device, do: :erlang.nif_error(:not_loaded)
  def set_default_device(_device), do: :erlang.nif_error(:not_loaded)
  def device_new(_type), do: :erlang.nif_error(:not_loaded)
  def synchronize(_stream), do: :erlang.nif_error(:not_loaded)
end
