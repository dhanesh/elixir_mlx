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

  @doc """
  Load weights from a file or directory, returning a flat map of `%{String.t() => Nx.Tensor.t()}`.

  Supports `.npy` files, `.safetensors` files, and directories containing
  `model.safetensors` or sharded safetensors with an index file.

  ## Options
    * `:format` - force format (`:npy` or `:safetensors`), auto-detected by default
    * `:dtype` - Nx type to cast all tensors to (e.g. `{:f, 16}`)

  ## Examples

      weights = Mlx.IO.load_weights("/path/to/model.safetensors")
      weights = Mlx.IO.load_weights("/path/to/model_dir")
      weights = Mlx.IO.load_weights("/path/to/array.npy")
  """
  def load_weights(path, opts \\ []) when is_binary(path) do
    format = Keyword.get(opts, :format)
    dtype = Keyword.get(opts, :dtype)

    tensors =
      cond do
        File.dir?(path) ->
          load_weights_from_dir(path)

        format == :npy || (!format && String.ends_with?(path, ".npy")) ->
          %{"weights" => load(path)}

        format == :safetensors || (!format && String.ends_with?(path, ".safetensors")) ->
          {tensors_map, _metadata} = load_safetensors(path)
          tensors_map

        true ->
          raise ArgumentError,
                "cannot detect format for #{path}; use :format option or a recognized extension (.npy, .safetensors)"
      end

    maybe_cast_dtype(tensors, dtype)
  end

  defp load_weights_from_dir(dir) do
    index_path = Path.join(dir, "model.safetensors.index.json")
    single_path = Path.join(dir, "model.safetensors")

    cond do
      File.exists?(index_path) ->
        load_sharded_safetensors(dir, index_path)

      File.exists?(single_path) ->
        {tensors_map, _metadata} = load_safetensors(single_path)
        tensors_map

      true ->
        raise ArgumentError,
              "directory #{dir} does not contain model.safetensors or model.safetensors.index.json"
    end
  end

  defp load_sharded_safetensors(dir, index_path) do
    index = index_path |> File.read!() |> Jason.decode!()
    weight_map = Map.get(index, "weight_map", %{})

    # Group weights by shard file — load each shard file only once
    shards_to_keys =
      Enum.group_by(weight_map, fn {_key, shard_file} -> shard_file end, fn {key, _shard} ->
        key
      end)

    Enum.reduce(shards_to_keys, %{}, fn {shard_file, keys}, acc ->
      shard_path = Path.join(dir, shard_file)
      {shard_tensors, _metadata} = load_safetensors(shard_path)

      Enum.reduce(keys, acc, fn key, inner_acc ->
        case Map.fetch(shard_tensors, key) do
          {:ok, tensor} -> Map.put(inner_acc, key, tensor)
          :error -> inner_acc
        end
      end)
    end)
  end

  defp maybe_cast_dtype(tensors, nil), do: tensors

  defp maybe_cast_dtype(tensors, dtype) do
    Map.new(tensors, fn {key, tensor} -> {key, Nx.as_type(tensor, dtype)} end)
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
