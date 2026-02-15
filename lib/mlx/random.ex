defmodule Mlx.Random do
  @moduledoc """
  Random number generation using MLX's PRNG.

  MLX uses a splittable PRNG (like JAX). You create a key with `key/1`,
  then split it for reproducible random sequences.

  ## Example

      key = Mlx.Random.key(42)
      {k1, k2} = Mlx.Random.split(key)
      t = Mlx.Random.normal(k1, {3, 3})
  """

  alias Mlx.NIF

  defp s, do: elem(NIF.default_cpu_stream(), 1)

  defp unwrap!({:ok, val}), do: val
  defp unwrap!({:error, msg}), do: raise("MLX error: #{msg}")

  defp to_mlx_scalar(val) do
    t = Nx.tensor(val, type: :f32, backend: Mlx.Backend)
    t.data.ref
  end

  @doc """
  Creates a PRNG key from an integer seed.

  Returns an opaque key tensor used for all random operations.
  """
  def key(seed) when is_integer(seed) do
    {:ok, ref} = NIF.mlx_random_key(seed)
    ref
  end

  @doc """
  Seeds the global random number generator.
  """
  def seed(seed) when is_integer(seed) do
    :ok = NIF.mlx_random_seed(seed)
    :ok
  end

  @doc """
  Splits a PRNG key into two new keys.

  Returns `{key1, key2}`.
  """
  def split(key) do
    {k1, k2} = unwrap!(NIF.mlx_random_split(key, s()))
    {k1, k2}
  end

  @doc """
  Splits a PRNG key into `num` new keys.

  Returns a single array of shape `{num, 2}` containing the keys.
  """
  def split(key, num) when is_integer(num) do
    unwrap!(NIF.mlx_random_split_num(key, num, s()))
  end

  @doc """
  Generates uniform random values in `[low, high)`.

  ## Options
    * `:shape` - output shape (default `{}` for scalar)
    * `:dtype` - output type (default `:f32`)
    * `:key` - PRNG key (default: global state)
  """
  def uniform(opts \\ []) do
    low = Keyword.get(opts, :low, 0.0)
    high = Keyword.get(opts, :high, 1.0)
    shape = Keyword.get(opts, :shape, {}) |> Tuple.to_list()
    dtype = Keyword.get(opts, :dtype, :f32)
    key = Keyword.get(opts, :key, nil)

    low_ref = to_mlx_scalar(low)
    high_ref = to_mlx_scalar(high)
    key_ref = key || nil

    ref = unwrap!(NIF.mlx_random_uniform(low_ref, high_ref, shape, dtype, key_ref, s()))

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: Mlx.Dtype.to_nx(dtype),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  @doc """
  Generates normal (Gaussian) random values.

  ## Options
    * `:shape` - output shape (default `{}` for scalar)
    * `:dtype` - output type (default `:f32`)
    * `:loc` - mean (default `0.0`)
    * `:scale` - standard deviation (default `1.0`)
    * `:key` - PRNG key (default: global state)
  """
  def normal(opts \\ []) do
    shape = Keyword.get(opts, :shape, {}) |> Tuple.to_list()
    dtype = Keyword.get(opts, :dtype, :f32)
    loc = Keyword.get(opts, :loc, 0.0) / 1
    scale = Keyword.get(opts, :scale, 1.0) / 1
    key = Keyword.get(opts, :key, nil)
    key_ref = key || nil

    ref = unwrap!(NIF.mlx_random_normal(shape, dtype, loc, scale, key_ref, s()))

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: Mlx.Dtype.to_nx(dtype),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  @doc """
  Generates Bernoulli random values (0 or 1).

  ## Options
    * `:p` - probability of 1 (default `0.5`, as a scalar Nx tensor)
    * `:shape` - output shape (default `{}`)
    * `:key` - PRNG key (default: global state)
  """
  def bernoulli(opts \\ []) do
    p = Keyword.get(opts, :p, 0.5)
    shape = Keyword.get(opts, :shape, {}) |> Tuple.to_list()
    key = Keyword.get(opts, :key, nil)

    p_ref = to_mlx_scalar(p)
    key_ref = key || nil

    ref = unwrap!(NIF.mlx_random_bernoulli(p_ref, shape, key_ref, s()))

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: {:u, 8},
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  @doc """
  Generates random integers in `[low, high)`.

  ## Options
    * `:shape` - output shape (default `{}`)
    * `:dtype` - output type (default `:s32`)
    * `:key` - PRNG key (default: global state)
  """
  def randint(low, high, opts \\ []) do
    shape = Keyword.get(opts, :shape, {}) |> Tuple.to_list()
    dtype = Keyword.get(opts, :dtype, :s32)
    key = Keyword.get(opts, :key, nil)

    low_ref = to_mlx_scalar(low)
    high_ref = to_mlx_scalar(high)
    key_ref = key || nil

    ref = unwrap!(NIF.mlx_random_randint(low_ref, high_ref, shape, dtype, key_ref, s()))

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: Mlx.Dtype.to_nx(dtype),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  @doc """
  Generates truncated normal random values in `[lower, upper]`.

  ## Options
    * `:shape` - output shape (default `{}`)
    * `:dtype` - output type (default `:f32`)
    * `:key` - PRNG key (default: global state)
  """
  def truncated_normal(lower, upper, opts \\ []) do
    shape = Keyword.get(opts, :shape, {}) |> Tuple.to_list()
    dtype = Keyword.get(opts, :dtype, :f32)
    key = Keyword.get(opts, :key, nil)

    lower_ref = to_mlx_scalar(lower)
    upper_ref = to_mlx_scalar(upper)
    key_ref = key || nil

    ref =
      unwrap!(NIF.mlx_random_truncated_normal(lower_ref, upper_ref, shape, dtype, key_ref, s()))

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: Mlx.Dtype.to_nx(dtype),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  @doc """
  Samples from a categorical distribution defined by logits.

  ## Options
    * `:axis` - axis along which to sample (default `-1`)
    * `:key` - PRNG key (default: global state)
  """
  def categorical(%Nx.Tensor{data: %Mlx.Backend{ref: logits_ref}} = logits, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    key = Keyword.get(opts, :key, nil)
    key_ref = key || nil

    ref = unwrap!(NIF.mlx_random_categorical(logits_ref, axis, key_ref, s()))

    # Output shape: logits shape with the sampled axis removed
    out_shape =
      logits.shape
      |> Tuple.to_list()
      |> List.delete_at(if(axis < 0, do: tuple_size(logits.shape) + axis, else: axis))
      |> List.to_tuple()

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: {:u, 32},
      shape: out_shape,
      names: List.duplicate(nil, tuple_size(out_shape))
    }
  end

  @doc """
  Generates Gumbel random values.

  ## Options
    * `:shape` - output shape (default `{}`)
    * `:dtype` - output type (default `:f32`)
    * `:key` - PRNG key (default: global state)
  """
  def gumbel(opts \\ []) do
    shape = Keyword.get(opts, :shape, {}) |> Tuple.to_list()
    dtype = Keyword.get(opts, :dtype, :f32)
    key = Keyword.get(opts, :key, nil)
    key_ref = key || nil

    ref = unwrap!(NIF.mlx_random_gumbel(shape, dtype, key_ref, s()))

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: Mlx.Dtype.to_nx(dtype),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  @doc """
  Generates Laplace-distributed random values.

  ## Options
    * `:shape` - output shape (default `{}`)
    * `:dtype` - output type (default `:f32`)
    * `:loc` - location parameter (default `0.0`)
    * `:scale` - scale parameter (default `1.0`)
    * `:key` - PRNG key (default: global state)
  """
  def laplace(opts \\ []) do
    shape = Keyword.get(opts, :shape, {}) |> Tuple.to_list()
    dtype = Keyword.get(opts, :dtype, :f32)
    loc = Keyword.get(opts, :loc, 0.0) / 1
    scale = Keyword.get(opts, :scale, 1.0) / 1
    key = Keyword.get(opts, :key, nil)
    key_ref = key || nil

    ref = unwrap!(NIF.mlx_random_laplace(shape, dtype, loc, scale, key_ref, s()))

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: Mlx.Dtype.to_nx(dtype),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end
end
