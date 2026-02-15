defmodule Mlx.Fast do
  @moduledoc """
  Performance-critical fused operations for transformer models.

  These operations are optimized for Apple Silicon via Metal and provide
  significant speedups over equivalent composed operations.

  ## Example

      x = Nx.tensor([[1.0, 2.0, 3.0]], backend: Mlx.Backend)
      weight = Nx.tensor([1.0, 1.0, 1.0], backend: Mlx.Backend)
      Mlx.Fast.rms_norm(x, weight, eps: 1.0e-5)
  """

  alias Mlx.NIF

  defp s, do: elem(NIF.default_gpu_stream(), 1)

  defp unwrap!({:ok, val}), do: val
  defp unwrap!({:error, msg}), do: raise("MLX error: #{msg}")

  defp from_ref(%Nx.Tensor{data: %Mlx.Backend{ref: ref}}), do: ref

  @doc """
  Layer normalization.

  Normalizes the input along the last axis, optionally applying learned
  weight and bias parameters.

  ## Options
    * `:eps` - epsilon for numerical stability (default: 1.0e-5)
  """
  def layer_norm(%Nx.Tensor{} = x, weight \\ nil, bias \\ nil, opts \\ []) do
    eps = Keyword.get(opts, :eps, 1.0e-5)
    w = if weight, do: from_ref(weight), else: nil
    b = if bias, do: from_ref(bias), else: nil
    ref = unwrap!(NIF.mlx_fast_layer_norm(from_ref(x), w, b, eps / 1, s()))
    to_nx_infer(ref)
  end

  @doc """
  Root mean square normalization.

  Normalizes the input by dividing by the RMS of the values along the
  last axis, then multiplying by the weight.

  ## Options
    * `:eps` - epsilon for numerical stability (default: 1.0e-5)
  """
  def rms_norm(%Nx.Tensor{} = x, %Nx.Tensor{} = weight, opts \\ []) do
    eps = Keyword.get(opts, :eps, 1.0e-5)
    ref = unwrap!(NIF.mlx_fast_rms_norm(from_ref(x), from_ref(weight), eps / 1, s()))
    to_nx_infer(ref)
  end

  @doc """
  Rotary Positional Encoding (RoPE).

  Applies rotary position embeddings to the input tensor.

  ## Options
    * `:dims` - number of dimensions to apply RoPE to (required)
    * `:traditional` - use traditional RoPE (default: false)
    * `:base` - base frequency (default: nil, uses MLX default 10000.0)
    * `:scale` - scaling factor (default: 1.0)
    * `:offset` - position offset (default: 0)
    * `:freqs` - custom frequency tensor (default: nil)
  """
  def rope(%Nx.Tensor{} = x, opts \\ []) do
    dims = Keyword.fetch!(opts, :dims)
    traditional = if Keyword.get(opts, :traditional, false), do: true, else: false
    base = Keyword.get(opts, :base, nil)
    scale = Keyword.get(opts, :scale, 1.0)
    offset = Keyword.get(opts, :offset, 0)
    freqs = Keyword.get(opts, :freqs, nil)

    base_val = if base, do: base / 1, else: nil
    freqs_ref = if freqs, do: from_ref(freqs), else: nil

    ref =
      unwrap!(
        NIF.mlx_fast_rope(
          from_ref(x),
          dims,
          traditional,
          base_val,
          scale / 1,
          offset,
          freqs_ref,
          s()
        )
      )

    to_nx_infer(ref)
  end

  @doc """
  Scaled Dot-Product Attention.

  Computes `softmax(Q @ K^T / scale) @ V` with optional mask,
  fused for efficiency on Apple Silicon.

  ## Arguments
    * `q` - query tensor (shape: [batch, heads, seq_len, head_dim])
    * `k` - key tensor
    * `v` - value tensor
    * `scale` - scaling factor (typically `1 / sqrt(head_dim)`)

  ## Options
    * `:mask` - optional attention mask tensor (default: nil)
  """
  def scaled_dot_product_attention(
        %Nx.Tensor{} = q,
        %Nx.Tensor{} = k,
        %Nx.Tensor{} = v,
        scale,
        opts \\ []
      ) do
    mask = Keyword.get(opts, :mask, nil)
    mask_ref = if mask, do: from_ref(mask), else: nil

    ref =
      unwrap!(
        NIF.mlx_fast_sdpa(
          from_ref(q),
          from_ref(k),
          from_ref(v),
          scale / 1,
          mask_ref,
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
