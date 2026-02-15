defmodule Mlx.FastTest do
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-4)
    a_list = Nx.to_flat_list(a)
    b_list = Nx.to_flat_list(b)

    Enum.zip(a_list, b_list)
    |> Enum.each(fn {av, bv} ->
      assert_in_delta(av, bv, atol, "expected #{av} to be close to #{bv}")
    end)
  end

  describe "layer_norm" do
    test "normalizes input with weight and bias" do
      x = Nx.tensor([[1.0, 2.0, 3.0]])
      weight = Nx.tensor([1.0, 1.0, 1.0])
      bias = Nx.tensor([0.0, 0.0, 0.0])
      result = Mlx.Fast.layer_norm(x, weight, bias, eps: 1.0e-5)
      # After normalization, mean should be ~0 and std ~1
      vals = Nx.to_flat_list(result)
      mean = Enum.sum(vals) / length(vals)
      assert_in_delta(mean, 0.0, 1.0e-4)
    end

    test "normalizes without weight and bias" do
      x = Nx.tensor([[1.0, 2.0, 3.0]])
      result = Mlx.Fast.layer_norm(x)
      vals = Nx.to_flat_list(result)
      mean = Enum.sum(vals) / length(vals)
      assert_in_delta(mean, 0.0, 1.0e-4)
    end
  end

  describe "rms_norm" do
    test "normalizes by root mean square" do
      x = Nx.tensor([[1.0, 2.0, 3.0]])
      weight = Nx.tensor([1.0, 1.0, 1.0])
      result = Mlx.Fast.rms_norm(x, weight, eps: 1.0e-5)
      # RMS norm divides by RMS, then multiplies by weight
      # RMS of [1,2,3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.1602
      rms = :math.sqrt((1.0 + 4.0 + 9.0) / 3.0)
      expected = Enum.map([1.0, 2.0, 3.0], &(&1 / rms))
      assert_all_close(result, Nx.tensor([expected]))
    end
  end

  describe "scaled_dot_product_attention" do
    test "basic attention without mask" do
      # Shape: [batch=1, heads=1, seq_len=2, head_dim=4]
      q = Nx.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]])
      k = Nx.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]])
      v = Nx.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])

      scale = 1.0 / :math.sqrt(4.0)
      result = Mlx.Fast.scaled_dot_product_attention(q, k, v, scale)
      assert Nx.shape(result) == {1, 1, 2, 4}
    end
  end

  describe "rope" do
    test "applies rotary positional encoding" do
      # Shape: [batch=1, seq_len=2, heads=1, head_dim=4]
      x = Nx.tensor([[[[1.0, 0.0, 1.0, 0.0]], [[1.0, 0.0, 1.0, 0.0]]]])
      result = Mlx.Fast.rope(x, dims: 4, offset: 0, scale: 1.0, base: 10000.0)
      assert Nx.shape(result) == {1, 2, 1, 4}
    end
  end
end
