defmodule Mlx.IO do
  @moduledoc """
  Array I/O operations for MLX.

  Supports saving/loading individual arrays (.npy format) and
  SafeTensors format for collections of named tensors.

  ## Example

      a = Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)
      Mlx.IO.save("/tmp/test.npy", a)
      loaded = Mlx.IO.load("/tmp/test.npy")
  """

  alias Mlx.NIF

  defp s, do: elem(NIF.default_cpu_stream(), 1)

  defp unwrap!({:ok, val}), do: val
  defp unwrap!({:error, msg}), do: raise("MLX error: #{msg}")

  defp from_ref(%Nx.Tensor{data: %Mlx.Backend{ref: ref}}), do: ref

  @doc """
  Save a single tensor to a .npy file.
  """
  def save(path, %Nx.Tensor{} = tensor) when is_binary(path) do
    ref = from_ref(tensor)
    NIF.eval(ref)
    case NIF.mlx_save(to_charlist(path), ref) do
      :ok -> :ok
      {:error, msg} -> raise "MLX error: #{msg}"
    end
  end

  @doc """
  Load a single tensor from a .npy file.
  """
  def load(path) when is_binary(path) do
    ref = unwrap!(NIF.mlx_load(to_charlist(path), s()))
    to_nx_infer(ref)
  end

  @doc """
  Save a map of named tensors in SafeTensors format.

  ## Arguments
    * `path` - file path (should end in `.safetensors`)
    * `tensors` - map of `%{String.t() => Nx.Tensor.t()}`

  ## Options
    * `:metadata` - optional map of string metadata (default: %{})
  """
  def save_safetensors(path, tensors, opts \\ []) when is_binary(path) and is_map(tensors) do
    metadata = Keyword.get(opts, :metadata, %{})

    {keys, arrays} =
      tensors
      |> Enum.map(fn {k, %Nx.Tensor{} = t} ->
        ref = from_ref(t)
        NIF.eval(ref)
        {to_charlist(k), ref}
      end)
      |> Enum.unzip()

    {meta_keys, meta_vals} =
      metadata
      |> Enum.map(fn {k, v} -> {to_charlist(k), to_charlist(v)} end)
      |> Enum.unzip()

    meta_keys = if meta_keys == [], do: [], else: meta_keys
    meta_vals = if meta_vals == [], do: [], else: meta_vals

    case NIF.mlx_save_safetensors(to_charlist(path), keys, arrays, meta_keys, meta_vals) do
      :ok -> :ok
      {:error, msg} -> raise "MLX error: #{msg}"
    end
  end

  @doc """
  Load tensors from a SafeTensors file.

  Returns `{tensors_map, metadata_map}` where:
    * `tensors_map` - `%{String.t() => Nx.Tensor.t()}`
    * `metadata_map` - `%{String.t() => String.t()}`
  """
  def load_safetensors(path) when is_binary(path) do
    {arr_pairs, meta_pairs} = unwrap!(NIF.mlx_load_safetensors(to_charlist(path), s()))

    tensors =
      arr_pairs
      |> Enum.map(fn {key, ref} ->
        {to_string(key), to_nx_infer(ref)}
      end)
      |> Map.new()

    metadata =
      meta_pairs
      |> Enum.map(fn {k, v} -> {to_string(k), to_string(v)} end)
      |> Map.new()

    {tensors, metadata}
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
