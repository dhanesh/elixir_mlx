defmodule Mlx.Quantize do
  @moduledoc """
  Weight quantization operations for MLX.

  Provides affine quantization for model compression on Apple Silicon.
  Supports 2, 4, and 8-bit quantization with configurable group sizes.

  ## Example

      w = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], backend: Mlx.Backend)
      {quantized, scales, biases} = Mlx.Quantize.quantize(w, group_size: 4, bits: 4)
      reconstructed = Mlx.Quantize.dequantize(quantized, scales, biases, group_size: 4, bits: 4)
  """

  alias Mlx.NIF

  defp s, do: elem(NIF.default_gpu_stream(), 1)

  defp unwrap!({:ok, val}), do: val
  defp unwrap!({:error, msg}), do: raise("MLX error: #{msg}")

  defp from_ref(%Nx.Tensor{data: %Mlx.Backend{ref: ref}}), do: ref

  @doc """
  Quantize a weight matrix.

  Returns `{quantized, scales, biases}` where the original weights can be
  approximately reconstructed as `dequantize(quantized, scales, biases)`.

  ## Options
    * `:group_size` - quantization group size (default: 64)
    * `:bits` - quantization bits, one of 2, 4, 8 (default: 4)
  """
  def quantize(%Nx.Tensor{} = w, opts \\ []) do
    group_size = Keyword.get(opts, :group_size, 64)
    bits = Keyword.get(opts, :bits, 4)
    {q_ref, s_ref, b_ref} = unwrap!(NIF.mlx_quantize(from_ref(w), group_size, bits, s()))
    {to_nx_infer(q_ref), to_nx_infer(s_ref), to_nx_infer(b_ref)}
  end

  @doc """
  Dequantize a quantized weight matrix.

  ## Options
    * `:group_size` - must match the value used during quantization (default: 64)
    * `:bits` - must match the value used during quantization (default: 4)
  """
  def dequantize(%Nx.Tensor{} = w, %Nx.Tensor{} = scales, %Nx.Tensor{} = biases, opts \\ []) do
    group_size = Keyword.get(opts, :group_size, 64)
    bits = Keyword.get(opts, :bits, 4)

    ref =
      unwrap!(
        NIF.mlx_dequantize(
          from_ref(w),
          from_ref(scales),
          from_ref(biases),
          group_size,
          bits,
          s()
        )
      )

    to_nx_infer(ref)
  end

  @doc """
  Perform matrix multiplication with a quantized weight matrix.

  Equivalent to `Nx.dot(x, dequantize(w, scales, biases))` but more efficient.

  ## Options
    * `:transpose` - whether to transpose w before multiplication (default: true)
    * `:group_size` - must match quantization (default: 64)
    * `:bits` - must match quantization (default: 4)
  """
  def quantized_matmul(
        %Nx.Tensor{} = x,
        %Nx.Tensor{} = w,
        %Nx.Tensor{} = scales,
        %Nx.Tensor{} = biases,
        opts \\ []
      ) do
    transpose = Keyword.get(opts, :transpose, true)
    group_size = Keyword.get(opts, :group_size, 64)
    bits = Keyword.get(opts, :bits, 4)

    ref =
      unwrap!(
        NIF.mlx_quantized_matmul(
          from_ref(x),
          from_ref(w),
          from_ref(scales),
          from_ref(biases),
          transpose,
          group_size,
          bits,
          s()
        )
      )

    to_nx_infer(ref)
  end

  # Helpers

  defp to_nx_infer(ref) do
    {:ok, shape_list} = NIF.shape(ref)
    {:ok, dtype_atom} = NIF.dtype(ref)

    type =
      case Mlx.Dtype.to_nx(dtype_atom) do
        nil -> {:f, 32}
        t -> t
      end

    shape = List.to_tuple(shape_list)

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: type,
      shape: shape,
      names: List.duplicate(nil, tuple_size(shape))
    }
  end
end
