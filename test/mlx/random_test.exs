defmodule Mlx.RandomTest do
  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  describe "key management" do
    test "key creates a PRNG key" do
      key = Mlx.Random.key(42)
      assert is_reference(key)
    end

    test "seed sets global seed" do
      assert :ok = Mlx.Random.seed(123)
    end

    test "split returns two keys" do
      key = Mlx.Random.key(42)
      {k1, k2} = Mlx.Random.split(key)
      assert is_reference(k1)
      assert is_reference(k2)
    end

    test "split with num returns array of keys" do
      key = Mlx.Random.key(42)
      keys = Mlx.Random.split(key, 4)
      assert is_reference(keys)
    end
  end

  describe "uniform" do
    test "generates uniform values with default shape" do
      t = Mlx.Random.uniform(shape: {3})
      assert Nx.shape(t) == {3}
      vals = Nx.to_flat_list(t)
      assert Enum.all?(vals, &(&1 >= 0.0 and &1 < 1.0))
    end

    test "generates uniform values with specified range" do
      t = Mlx.Random.uniform(low: 2.0, high: 5.0, shape: {100})
      vals = Nx.to_flat_list(t)
      assert Enum.all?(vals, &(&1 >= 2.0 and &1 < 5.0))
    end

    test "uniform with key is reproducible" do
      key = Mlx.Random.key(42)
      {k1, _k2} = Mlx.Random.split(key)
      t1 = Mlx.Random.uniform(shape: {5}, key: k1)
      # Same key should produce same values
      t2 = Mlx.Random.uniform(shape: {5}, key: k1)
      assert Nx.to_flat_list(t1) == Nx.to_flat_list(t2)
    end
  end

  describe "normal" do
    test "generates normal values with default params" do
      t = Mlx.Random.normal(shape: {1000})
      assert Nx.shape(t) == {1000}
      # Mean should be approximately 0
      mean = Nx.to_number(Nx.mean(t))
      assert_in_delta(mean, 0.0, 0.2)
    end

    test "generates normal with loc and scale" do
      t = Mlx.Random.normal(shape: {1000}, loc: 5.0, scale: 0.1)
      mean = Nx.to_number(Nx.mean(t))
      assert_in_delta(mean, 5.0, 0.1)
    end
  end

  describe "bernoulli" do
    test "generates bernoulli values" do
      t = Mlx.Random.bernoulli(shape: {100}, p: 0.5)
      assert Nx.shape(t) == {100}
      vals = Nx.to_flat_list(t)
      assert Enum.all?(vals, &(&1 == 0 or &1 == 1))
    end
  end

  describe "randint" do
    test "generates random integers in range" do
      t = Mlx.Random.randint(0, 10, shape: {100})
      assert Nx.shape(t) == {100}
      vals = Nx.to_flat_list(t)
      assert Enum.all?(vals, &(&1 >= 0 and &1 < 10))
    end
  end

  describe "truncated_normal" do
    test "generates values within bounds" do
      t = Mlx.Random.truncated_normal(-2.0, 2.0, shape: {1000})
      assert Nx.shape(t) == {1000}
      vals = Nx.to_flat_list(t)
      assert Enum.all?(vals, &(&1 >= -2.0 and &1 <= 2.0))
    end
  end

  describe "gumbel" do
    test "generates gumbel values" do
      t = Mlx.Random.gumbel(shape: {100})
      assert Nx.shape(t) == {100}
      assert Nx.type(t) == {:f, 32}
    end
  end

  describe "laplace" do
    test "generates laplace values" do
      t = Mlx.Random.laplace(shape: {1000})
      assert Nx.shape(t) == {1000}
      mean = Nx.to_number(Nx.mean(t))
      assert_in_delta(mean, 0.0, 0.2)
    end
  end

  describe "categorical" do
    test "samples from categorical distribution" do
      # 3 categories, biased toward last
      logits = Nx.tensor([[0.0, 0.0, 10.0]])
      t = Mlx.Random.categorical(logits)
      # Should almost always pick category 2
      val = Nx.to_flat_list(t)
      assert hd(val) == 2
    end
  end
end
