defmodule Mlx.FFT do
  @moduledoc """
  Fast Fourier Transform operations using MLX.

  Provides 1D, 2D, and N-dimensional FFTs (complex and real) accelerated
  on Apple Silicon via Metal.

  ## Example

      a = Nx.tensor([1.0, 0.0, 0.0, 0.0], backend: Mlx.Backend)
      freq = Mlx.FFT.fft(a)
  """

  alias Mlx.NIF

  defp s, do: elem(NIF.default_cpu_stream(), 1)

  defp unwrap!({:ok, val}), do: val
  defp unwrap!({:error, msg}), do: raise("MLX error: #{msg}")

  defp from_ref(%Nx.Tensor{data: %Mlx.Backend{ref: ref}}), do: ref

  defp resolve_axis(ax, ndim) when ax < 0, do: ndim + ax
  defp resolve_axis(ax, _ndim), do: ax

  @doc """
  1D discrete Fourier transform.

  ## Options
    * `:n` - transform length (default: axis size)
    * `:axis` - axis to transform (default: -1)
  """
  def fft(%Nx.Tensor{} = a, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    resolved_axis = resolve_axis(axis, tuple_size(a.shape))
    n = Keyword.get(opts, :n, elem(a.shape, resolved_axis))
    ref = unwrap!(NIF.mlx_fft(from_ref(a), n, axis, s()))
    to_nx_infer(ref)
  end

  @doc """
  1D inverse discrete Fourier transform.
  """
  def ifft(%Nx.Tensor{} = a, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    resolved_axis = resolve_axis(axis, tuple_size(a.shape))
    n = Keyword.get(opts, :n, elem(a.shape, resolved_axis))
    ref = unwrap!(NIF.mlx_ifft(from_ref(a), n, axis, s()))
    to_nx_infer(ref)
  end

  @doc """
  1D real-input FFT.

  Input must be real-valued. Output has shape `n/2 + 1` along the transform axis.
  """
  def rfft(%Nx.Tensor{} = a, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    resolved_axis = resolve_axis(axis, tuple_size(a.shape))
    n = Keyword.get(opts, :n, elem(a.shape, resolved_axis))
    ref = unwrap!(NIF.mlx_rfft(from_ref(a), n, axis, s()))
    to_nx_infer(ref)
  end

  @doc """
  1D inverse real FFT.

  For input of length `m` along the transform axis, output length is `2*(m-1)`.
  Use `:n` to specify the exact output length.
  """
  def irfft(%Nx.Tensor{} = a, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    resolved_axis = resolve_axis(axis, tuple_size(a.shape))
    # irfft: input has n/2+1 elements, output has n elements
    # default n = 2 * (input_size - 1)
    n = Keyword.get(opts, :n, 2 * (elem(a.shape, resolved_axis) - 1))
    ref = unwrap!(NIF.mlx_irfft(from_ref(a), n, axis, s()))
    to_nx_infer(ref)
  end

  @doc """
  2D discrete Fourier transform.

  ## Options
    * `:n` - list of transform lengths (default: axis sizes)
    * `:axes` - list of axes to transform (default: last 2)
  """
  def fft2(%Nx.Tensor{} = a, opts \\ []) do
    {n_list, axes_list} = multi_opts(a, 2, opts)
    ref = unwrap!(NIF.mlx_fft2(from_ref(a), n_list, axes_list, s()))
    to_nx_infer(ref)
  end

  @doc """
  2D inverse discrete Fourier transform.
  """
  def ifft2(%Nx.Tensor{} = a, opts \\ []) do
    {n_list, axes_list} = multi_opts(a, 2, opts)
    ref = unwrap!(NIF.mlx_ifft2(from_ref(a), n_list, axes_list, s()))
    to_nx_infer(ref)
  end

  @doc """
  N-dimensional discrete Fourier transform.
  """
  def fftn(%Nx.Tensor{} = a, opts \\ []) do
    ndim = tuple_size(a.shape)
    {n_list, axes_list} = multi_opts(a, ndim, opts)
    ref = unwrap!(NIF.mlx_fftn(from_ref(a), n_list, axes_list, s()))
    to_nx_infer(ref)
  end

  @doc """
  N-dimensional inverse discrete Fourier transform.
  """
  def ifftn(%Nx.Tensor{} = a, opts \\ []) do
    ndim = tuple_size(a.shape)
    {n_list, axes_list} = multi_opts(a, ndim, opts)
    ref = unwrap!(NIF.mlx_ifftn(from_ref(a), n_list, axes_list, s()))
    to_nx_infer(ref)
  end

  @doc """
  2D real-input FFT.
  """
  def rfft2(%Nx.Tensor{} = a, opts \\ []) do
    {n_list, axes_list} = multi_opts(a, 2, opts)
    ref = unwrap!(NIF.mlx_rfft2(from_ref(a), n_list, axes_list, s()))
    to_nx_infer(ref)
  end

  @doc """
  2D inverse real FFT.

  Use `:n` to specify the output shape along the transform axes.
  The last axis defaults to `2*(input_size - 1)`.
  """
  def irfft2(%Nx.Tensor{} = a, opts \\ []) do
    {n_list, axes_list} = multi_opts_irfft(a, 2, opts)
    ref = unwrap!(NIF.mlx_irfft2(from_ref(a), n_list, axes_list, s()))
    to_nx_infer(ref)
  end

  @doc """
  N-dimensional real-input FFT.
  """
  def rfftn(%Nx.Tensor{} = a, opts \\ []) do
    ndim = tuple_size(a.shape)
    {n_list, axes_list} = multi_opts(a, ndim, opts)
    ref = unwrap!(NIF.mlx_rfftn(from_ref(a), n_list, axes_list, s()))
    to_nx_infer(ref)
  end

  @doc """
  N-dimensional inverse real FFT.

  Use `:n` to specify the output shape along the transform axes.
  The last axis defaults to `2*(input_size - 1)`.
  """
  def irfftn(%Nx.Tensor{} = a, opts \\ []) do
    ndim = tuple_size(a.shape)
    {n_list, axes_list} = multi_opts_irfft(a, ndim, opts)
    ref = unwrap!(NIF.mlx_irfftn(from_ref(a), n_list, axes_list, s()))
    to_nx_infer(ref)
  end

  # Helpers

  defp multi_opts(tensor, default_ndim, opts) do
    axes = Keyword.get(opts, :axes, nil)
    axes_list = axes || Enum.to_list(-(min(default_ndim, tuple_size(tensor.shape)))..-1)

    n = Keyword.get(opts, :n, nil)
    n_list = n || Enum.map(axes_list, fn ax ->
      resolved = resolve_axis(ax, tuple_size(tensor.shape))
      elem(tensor.shape, resolved)
    end)

    {n_list, axes_list}
  end

  # For inverse real FFTs: last axis size defaults to 2*(input_size - 1)
  defp multi_opts_irfft(tensor, default_ndim, opts) do
    axes = Keyword.get(opts, :axes, nil)
    axes_list = axes || Enum.to_list(-(min(default_ndim, tuple_size(tensor.shape)))..-1)

    n = Keyword.get(opts, :n, nil)
    n_list = n || axes_list
      |> Enum.with_index()
      |> Enum.map(fn {ax, idx} ->
        resolved = resolve_axis(ax, tuple_size(tensor.shape))
        size = elem(tensor.shape, resolved)
        # Last axis in irfft: output = 2*(input-1)
        if idx == length(axes_list) - 1 do
          2 * (size - 1)
        else
          size
        end
      end)

    {n_list, axes_list}
  end

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
