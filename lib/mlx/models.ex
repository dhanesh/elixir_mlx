defmodule Mlx.Models do
  @moduledoc """
  High-level model loading pipeline for MLX.

  Provides a complete pipeline from model files (local or HuggingFace Hub)
  to `Axon.ModelState` structs, including weight unflattening, optional
  dequantization, and key remapping.

  ## Example

      # Load from a local directory
      state = Mlx.Models.load_model_state("/path/to/model_dir")

      # Load with dtype casting
      state = Mlx.Models.load_model_state("/path/to/model_dir", dtype: {:f, 16})

      # Load from HuggingFace Hub (requires :req dep)
      state = Mlx.Models.load_model_state("bert-base-uncased")
  """

  @doc """
  Load a model configuration from a `config.json` file.

  Accepts either a directory path (looks for `config.json` inside) or
  a direct path to a JSON file.

  Returns a map of the parsed JSON.
  """
  def load_config(path) when is_binary(path) do
    config_path =
      if File.dir?(path) do
        Path.join(path, "config.json")
      else
        path
      end

    unless File.exists?(config_path) do
      raise ArgumentError, "config file not found: #{config_path}"
    end

    config_path |> File.read!() |> Jason.decode!()
  end

  @doc """
  Unflatten a map with dot-separated keys into a nested map.

  ## Example

      iex> Mlx.Models.unflatten_params(%{"a.b.c" => 1, "a.b.d" => 2, "x" => 3})
      %{"a" => %{"b" => %{"c" => 1, "d" => 2}}, "x" => 3}
  """
  def unflatten_params(flat_map) when is_map(flat_map) do
    Enum.reduce(flat_map, %{}, fn {key, value}, acc ->
      parts = String.split(key, ".")
      deep_put(acc, parts, value)
    end)
  end

  @doc """
  Create an `Axon.ModelState` from a weight map.

  ## Options
    * `:unflatten` - unflatten dot-separated keys into nested map (default: `true`)
    * `:remap_fn` - function `(key :: String.t()) -> String.t()` to remap weight names
  """
  def to_model_state(weights, opts \\ []) when is_map(weights) do
    ensure_axon!()
    unflatten = Keyword.get(opts, :unflatten, true)
    remap_fn = Keyword.get(opts, :remap_fn)

    weights =
      if remap_fn do
        Map.new(weights, fn {k, v} -> {remap_fn.(k), v} end)
      else
        weights
      end

    weights =
      if unflatten do
        unflatten_params(weights)
      else
        weights
      end

    Axon.ModelState.new(weights)
  end

  @doc """
  Full model loading pipeline: resolve path → load weights → dequantize → remap → unflatten → ModelState.

  Accepts a local directory path, a local file path, or a HuggingFace repo ID.

  ## Options
    * `:revision` - HF revision (default: `"main"`)
    * `:token` - HF API token
    * `:cache_dir` - override Hub cache directory
    * `:dtype` - Nx type to cast all tensors to
    * `:remap_fn` - function to remap weight names
    * `:unflatten` - unflatten dot-separated keys (default: `true`)
    * `:dequantize` - dequantize quantized weights if `quantize_config.json` exists (default: `true`)
  """
  def load_model_state(path_or_repo_id, opts \\ []) do
    ensure_axon!()
    dtype = Keyword.get(opts, :dtype)
    remap_fn = Keyword.get(opts, :remap_fn)
    unflatten = Keyword.get(opts, :unflatten, true)
    do_dequantize = Keyword.get(opts, :dequantize, true)

    model_dir = resolve_model_dir(path_or_repo_id, opts)

    # Load weights
    weights = Mlx.IO.load_weights(model_dir, dtype: dtype)

    # Dequantize if quantize_config.json exists
    weights =
      if do_dequantize do
        quantize_config_path = Path.join(model_dir, "quantize_config.json")

        if File.exists?(quantize_config_path) do
          dequantize_weights(weights, quantize_config_path)
        else
          weights
        end
      else
        weights
      end

    # Remap keys
    weights =
      if remap_fn do
        Map.new(weights, fn {k, v} -> {remap_fn.(k), v} end)
      else
        weights
      end

    # Unflatten
    weights =
      if unflatten do
        unflatten_params(weights)
      else
        weights
      end

    Axon.ModelState.new(weights)
  end

  # Private helpers

  defp resolve_model_dir(path_or_repo_id, opts) do
    cond do
      File.dir?(path_or_repo_id) ->
        path_or_repo_id

      File.exists?(path_or_repo_id) ->
        # Single file — return its parent directory
        Path.dirname(path_or_repo_id)

      true ->
        # Assume it's a HuggingFace repo ID — download via Hub
        hub_opts =
          opts
          |> Keyword.take([:revision, :token, :cache_dir, :force_download])
          |> Keyword.put(:allow_patterns, ["*.safetensors", "*.json", "*.safetensors.index.json"])

        Mlx.Hub.snapshot_download(path_or_repo_id, hub_opts)
    end
  end

  defp dequantize_weights(weights, config_path) do
    config = config_path |> File.read!() |> Jason.decode!()
    group_size = Map.get(config, "group_size", 64)
    bits = Map.get(config, "bits", 4)

    # Find weight/scales/biases triplets:
    # For a quantized layer "foo.weight", there will be "foo.scales" and "foo.biases"
    weight_keys =
      weights
      |> Map.keys()
      |> Enum.filter(&String.ends_with?(&1, ".weight"))

    Enum.reduce(weight_keys, weights, fn weight_key, acc ->
      base = String.replace_suffix(weight_key, ".weight", "")
      scales_key = "#{base}.scales"
      biases_key = "#{base}.biases"

      case {Map.fetch(acc, scales_key), Map.fetch(acc, biases_key)} do
        {{:ok, scales}, {:ok, biases}} ->
          dequantized =
            Mlx.Quantize.dequantize(acc[weight_key], scales, biases,
              group_size: group_size,
              bits: bits
            )

          acc
          |> Map.put(weight_key, dequantized)
          |> Map.delete(scales_key)
          |> Map.delete(biases_key)

        _ ->
          acc
      end
    end)
  end

  defp deep_put(map, [key], value) do
    Map.put(map, key, value)
  end

  defp deep_put(map, [key | rest], value) do
    existing = Map.get(map, key, %{})
    Map.put(map, key, deep_put(existing, rest, value))
  end

  defp ensure_axon! do
    unless Code.ensure_loaded?(Axon.ModelState) do
      raise """
      Axon is required for model state operations.
      Add {:axon, "~> 0.8"} to your dependencies in mix.exs.
      """
    end
  end
end
