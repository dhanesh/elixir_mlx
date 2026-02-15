defmodule Mlx.NNTest do
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

  # ── Activation Tests ──────────────────────────────────

  describe "relu" do
    test "zeros negative values, passes positive" do
      x = Nx.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
      result = Mlx.NN.relu(x)
      assert_all_close(result, Nx.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    end

    test "all zeros for negative input" do
      x = Nx.tensor([-5.0, -3.0, -1.0])
      result = Mlx.NN.relu(x)
      assert_all_close(result, Nx.tensor([0.0, 0.0, 0.0]))
    end
  end

  describe "relu6" do
    test "clips between 0 and 6" do
      x = Nx.tensor([-1.0, 0.0, 3.0, 6.0, 10.0])
      result = Mlx.NN.relu6(x)
      assert_all_close(result, Nx.tensor([0.0, 0.0, 3.0, 6.0, 6.0]))
    end
  end

  describe "leaky_relu" do
    test "default alpha=0.01" do
      x = Nx.tensor([-2.0, 0.0, 2.0])
      result = Mlx.NN.leaky_relu(x)
      assert_all_close(result, Nx.tensor([-0.02, 0.0, 2.0]))
    end

    test "custom alpha" do
      x = Nx.tensor([-1.0, 1.0])
      result = Mlx.NN.leaky_relu(x, alpha: 0.1)
      assert_all_close(result, Nx.tensor([-0.1, 1.0]))
    end
  end

  describe "elu" do
    test "default alpha=1.0" do
      x = Nx.tensor([-1.0, 0.0, 1.0])
      result = Mlx.NN.elu(x)
      # elu(-1) = 1.0 * (exp(-1) - 1) ≈ -0.6321
      assert_all_close(result, Nx.tensor([:math.exp(-1) - 1, 0.0, 1.0]))
    end

    test "custom alpha" do
      x = Nx.tensor([-1.0, 1.0])
      result = Mlx.NN.elu(x, alpha: 2.0)
      assert_all_close(result, Nx.tensor([2.0 * (:math.exp(-1) - 1), 1.0]))
    end
  end

  describe "selu" do
    test "scaled ELU with fixed constants" do
      x = Nx.tensor([0.0, 1.0])
      result = Mlx.NN.selu(x)
      lambda = 1.0507009873554805
      # selu(0) = lambda * 0 = 0
      # selu(1) = lambda * 1
      assert_all_close(result, Nx.tensor([0.0, lambda]))
    end
  end

  describe "gelu" do
    test "known values" do
      x = Nx.tensor([0.0, 1.0, -1.0])
      result = Mlx.NN.gelu(x)
      # gelu(0) = 0
      # gelu(1) ≈ 0.8413
      # gelu(-1) ≈ -0.1587
      expected_1 = 0.5 * (1 + :math.erf(1 / :math.sqrt(2)))
      expected_neg1 = -1 * 0.5 * (1 + :math.erf(-1 / :math.sqrt(2)))
      assert_all_close(result, Nx.tensor([0.0, expected_1, expected_neg1]), atol: 1.0e-3)
    end
  end

  describe "silu/swish" do
    test "silu = x * sigmoid(x)" do
      x = Nx.tensor([0.0, 1.0, -1.0])
      result = Mlx.NN.silu(x)
      # silu(0) = 0 * 0.5 = 0
      # silu(1) = 1 * sigmoid(1) ≈ 0.7311
      sig_1 = 1.0 / (1.0 + :math.exp(-1.0))
      sig_neg1 = 1.0 / (1.0 + :math.exp(1.0))
      assert_all_close(result, Nx.tensor([0.0, sig_1, -sig_neg1]), atol: 1.0e-3)
    end

    test "swish is alias for silu" do
      x = Nx.tensor([1.0, 2.0])
      assert_all_close(Mlx.NN.silu(x), Mlx.NN.swish(x))
    end
  end

  describe "mish" do
    test "mish = x * tanh(softplus(x))" do
      x = Nx.tensor([0.0, 1.0])
      result = Mlx.NN.mish(x)
      # mish(0) = 0 * tanh(ln(2)) ≈ 0
      sp_1 = :math.log(1 + :math.exp(1))
      mish_1 = 1.0 * :math.tanh(sp_1)
      assert_all_close(result, Nx.tensor([0.0, mish_1]), atol: 1.0e-3)
    end
  end

  describe "softplus" do
    test "default beta=1.0" do
      x = Nx.tensor([0.0, 1.0])
      result = Mlx.NN.softplus(x)
      assert_all_close(result, Nx.tensor([:math.log(2.0), :math.log(1 + :math.exp(1.0))]))
    end

    test "custom beta" do
      x = Nx.tensor([0.0])
      result = Mlx.NN.softplus(x, beta: 2.0)
      # log(1 + exp(2*0)) / 2 = log(2) / 2
      assert_all_close(result, Nx.tensor([:math.log(2.0) / 2]))
    end
  end

  describe "celu" do
    test "default alpha=1.0 matches elu" do
      x = Nx.tensor([-1.0, 0.0, 1.0])
      assert_all_close(Mlx.NN.celu(x), Mlx.NN.elu(x), atol: 1.0e-3)
    end
  end

  describe "log_sigmoid" do
    test "known values" do
      x = Nx.tensor([0.0, 10.0, -10.0])
      result = Mlx.NN.log_sigmoid(x)
      # log_sigmoid(0) = log(0.5) ≈ -0.6931
      # log_sigmoid(10) ≈ 0 (close to 0 for large positive)
      # log_sigmoid(-10) ≈ -10 (close to -x for large negative)
      assert_all_close(result, Nx.tensor([:math.log(0.5), -4.54e-5, -10.0]), atol: 1.0e-3)
    end
  end

  describe "hard_sigmoid" do
    test "piecewise linear approximation" do
      x = Nx.tensor([-4.0, 0.0, 4.0])
      result = Mlx.NN.hard_sigmoid(x)
      # hard_sigmoid(-4) = clip(-4/6 + 0.5, 0, 1) = 0
      # hard_sigmoid(0) = clip(0.5, 0, 1) = 0.5
      # hard_sigmoid(4) = clip(4/6 + 0.5, 0, 1) = 1
      assert_all_close(result, Nx.tensor([0.0, 0.5, 1.0]), atol: 0.05)
    end
  end

  describe "hard_swish" do
    test "x * hard_sigmoid(x)" do
      x = Nx.tensor([0.0, 3.0])
      result = Mlx.NN.hard_swish(x)
      # hard_swish(0) = 0 * 0.5 = 0
      # hard_swish(3) = 3 * clip(3/6 + 0.5, 0, 1) = 3 * 1 = 3
      assert_all_close(result, Nx.tensor([0.0, 3.0]))
    end
  end

  # ── Loss Function Tests ──────────────────────────────

  describe "mse_loss" do
    test "zero loss for identical tensors" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      assert_all_close(Mlx.NN.mse_loss(a, a), Nx.tensor(0.0))
    end

    test "correct mean squared error" do
      pred = Nx.tensor([1.0, 2.0, 3.0])
      target = Nx.tensor([1.0, 3.0, 5.0])
      # MSE = mean([0, 1, 4]) = 5/3 ≈ 1.6667
      assert_all_close(Mlx.NN.mse_loss(pred, target), Nx.tensor(5.0 / 3), atol: 1.0e-3)
    end
  end

  describe "l1_loss" do
    test "mean absolute error" do
      pred = Nx.tensor([1.0, 2.0, 3.0])
      target = Nx.tensor([1.0, 3.0, 5.0])
      # L1 = mean([0, 1, 2]) = 1.0
      assert_all_close(Mlx.NN.l1_loss(pred, target), Nx.tensor(1.0))
    end
  end

  describe "huber_loss" do
    test "quadratic for small errors, linear for large" do
      pred = Nx.tensor([0.0, 0.0])
      target = Nx.tensor([0.5, 2.0])
      # error=0.5 (< delta=1): 0.5 * 0.5^2 = 0.125
      # error=2.0 (> delta=1): 1*2 - 0.5*1 = 1.5
      # mean = (0.125 + 1.5) / 2 = 0.8125
      assert_all_close(Mlx.NN.huber_loss(pred, target), Nx.tensor(0.8125), atol: 1.0e-3)
    end
  end

  describe "binary_cross_entropy" do
    test "perfect predictions yield near-zero loss" do
      pred = Nx.tensor([0.99, 0.01])
      target = Nx.tensor([1.0, 0.0])
      loss = Mlx.NN.binary_cross_entropy(pred, target)
      assert Nx.to_number(loss) < 0.05
    end

    test "bad predictions yield high loss" do
      pred = Nx.tensor([0.1, 0.9])
      target = Nx.tensor([1.0, 0.0])
      loss = Mlx.NN.binary_cross_entropy(pred, target)
      assert Nx.to_number(loss) > 1.0
    end
  end

  describe "cross_entropy" do
    test "one-hot targets" do
      logits = Nx.tensor([[2.0, 1.0, 0.1]])
      targets = Nx.tensor([[1.0, 0.0, 0.0]])
      loss = Mlx.NN.cross_entropy(logits, targets)
      # Should be positive and small since logit[0] is highest
      assert Nx.to_number(loss) > 0
      assert Nx.to_number(loss) < 1.0
    end

    test "with label smoothing" do
      logits = Nx.tensor([[2.0, 1.0, 0.1]])
      targets = Nx.tensor([[1.0, 0.0, 0.0]])
      loss_no_smooth = Mlx.NN.cross_entropy(logits, targets)
      loss_smooth = Mlx.NN.cross_entropy(logits, targets, label_smoothing: 0.1)
      # Smoothed loss should differ from unsmoothed
      refute_in_delta Nx.to_number(loss_no_smooth), Nx.to_number(loss_smooth), 1.0e-5
    end
  end

  # ── Functional Layer Tests ──────────────────────────

  describe "linear" do
    test "x @ weight^T + bias" do
      x = Nx.tensor([[1.0, 2.0]])
      weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
      bias = Nx.tensor([0.1, 0.2, 0.3])
      result = Mlx.NN.linear(x, weight, bias)
      # [1,2] @ [[1,0],[0,1],[1,1]]^T = [1,2] @ [[1,0,1],[0,1,1]] = [1, 2, 3]
      # + [0.1, 0.2, 0.3] = [1.1, 2.2, 3.3]
      assert_all_close(result, Nx.tensor([[1.1, 2.2, 3.3]]))
    end

    test "without bias (zero bias is no-op)" do
      x = Nx.tensor([[1.0, 0.0]])
      weight = Nx.tensor([[2.0, 0.0]])
      result = Mlx.NN.linear(x, weight)
      assert_all_close(result, Nx.tensor([[2.0]]))
    end
  end

  describe "embedding" do
    test "lookup by indices" do
      table = Nx.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
      indices = Nx.tensor([0, 2, 1])
      result = Mlx.NN.embedding(table, indices)
      assert_all_close(result, Nx.tensor([[0.1, 0.2], [0.5, 0.6], [0.3, 0.4]]))
    end
  end

  describe "dropout" do
    test "training mode zeros some elements" do
      Mlx.Random.seed(42)
      x = Nx.broadcast(Nx.tensor(1.0), {100})
      result = Mlx.NN.dropout(x, rate: 0.5, training: true)
      vals = Nx.to_flat_list(result)
      zeros = Enum.count(vals, &(&1 == 0.0))
      # With p=0.5 and 100 elements, expect ~50 zeros (allow wide tolerance)
      assert zeros > 20
      assert zeros < 80
      # Non-zero values should be scaled by 1/(1-rate) = 2
      nonzero = Enum.filter(vals, &(&1 != 0.0))
      Enum.each(nonzero, fn v -> assert_in_delta(v, 2.0, 0.1) end)
    end

    test "eval mode is identity" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      result = Mlx.NN.dropout(x, rate: 0.5, training: false)
      assert_all_close(result, x)
    end
  end

  # ── Re-export Tests ──────────────────────────────────

  describe "re-exports" do
    test "layer_norm delegates to Mlx.Fast" do
      x = Nx.tensor([[1.0, 2.0, 3.0]])
      weight = Nx.tensor([1.0, 1.0, 1.0])
      result = Mlx.NN.layer_norm(x, weight)
      expected = Mlx.Fast.layer_norm(x, weight)
      assert_all_close(result, expected)
    end

    test "rms_norm delegates to Mlx.Fast" do
      x = Nx.tensor([[1.0, 2.0, 3.0]])
      weight = Nx.tensor([1.0, 1.0, 1.0])
      result = Mlx.NN.rms_norm(x, weight)
      expected = Mlx.Fast.rms_norm(x, weight)
      assert_all_close(result, expected)
    end
  end
end
