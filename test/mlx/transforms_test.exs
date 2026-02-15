defmodule Mlx.TransformsTest do
  use ExUnit.Case, async: false

  @moduletag :transforms

  describe "Mlx.Transforms.value_and_grad" do
    test "scalar quadratic: f(x) = x^2" do
      vag_fn = Mlx.Transforms.value_and_grad(fn [x] ->
        [Nx.pow(x, 2)]
      end)

      {[value], [grad]} = vag_fn.([Nx.tensor(3.0, backend: Mlx.Backend)])

      assert_in_delta Nx.to_number(value), 9.0, 1.0e-5
      assert_in_delta Nx.to_number(grad), 6.0, 1.0e-5
    end

    test "vector sum of squares: f(x) = sum(x^2)" do
      vag_fn = Mlx.Transforms.value_and_grad(fn [x] ->
        [Nx.sum(Nx.pow(x, 2))]
      end)

      {[value], [grad]} = vag_fn.([Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)])

      assert_in_delta Nx.to_number(value), 14.0, 1.0e-5
      # grad of sum(x^2) = 2x
      expected = [2.0, 4.0, 6.0]
      for {actual, exp} <- Enum.zip(Nx.to_flat_list(grad), expected) do
        assert_in_delta actual, exp, 1.0e-5
      end
    end

    test "multi-argument: f(x, y) = sum(x * y), grad w.r.t. x" do
      vag_fn = Mlx.Transforms.value_and_grad(fn [x, y] ->
        [Nx.sum(Nx.multiply(x, y))]
      end, [0])

      x = Nx.tensor([1.0, 2.0], backend: Mlx.Backend)
      y = Nx.tensor([3.0, 4.0], backend: Mlx.Backend)
      {[value], [grad_x]} = vag_fn.([x, y])

      # sum(x * y) = 1*3 + 2*4 = 11
      assert_in_delta Nx.to_number(value), 11.0, 1.0e-5
      # d/dx sum(x * y) = y
      assert Nx.to_flat_list(grad_x) == [3.0, 4.0]
    end

    test "sigmoid loss" do
      vag_fn = Mlx.Transforms.value_and_grad(fn [x] ->
        [Nx.sum(Nx.sigmoid(x))]
      end)

      {[value], [grad]} = vag_fn.([Nx.tensor([0.0], backend: Mlx.Backend)])

      # sigmoid(0) = 0.5
      assert_in_delta Nx.to_number(value), 0.5, 1.0e-5
      # sigmoid'(0) = 0.25
      assert_in_delta hd(Nx.to_flat_list(grad)), 0.25, 1.0e-5
    end
  end

  describe "Mlx.Transforms.grad" do
    test "simple gradient" do
      grad_fn = Mlx.Transforms.grad(fn [x] ->
        [Nx.sum(Nx.pow(x, 2))]
      end)

      [grad] = grad_fn.([Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)])

      expected = [2.0, 4.0, 6.0]
      for {actual, exp} <- Enum.zip(Nx.to_flat_list(grad), expected) do
        assert_in_delta actual, exp, 1.0e-5
      end
    end

    test "gradient at zero" do
      grad_fn = Mlx.Transforms.grad(fn [x] ->
        [Nx.pow(x, 2)]
      end)

      [grad] = grad_fn.([Nx.tensor(0.0, backend: Mlx.Backend)])
      assert_in_delta Nx.to_number(grad), 0.0, 1.0e-5
    end
  end

  describe "Mlx.Transforms.vjp" do
    test "vector-Jacobian product of identity" do
      # f(x) = x, so J = I, and vjp = cotangent * I = cotangent
      {_outputs, vjps} = Mlx.Transforms.vjp(
        fn [x] -> [x] end,
        [Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)],
        [Nx.tensor([1.0, 1.0, 1.0], backend: Mlx.Backend)]
      )

      [vjp] = vjps
      assert Nx.to_flat_list(vjp) == [1.0, 1.0, 1.0]
    end

    test "vjp of scalar multiply" do
      # f(x) = 2*x, J = 2I, vjp = 2 * cotangent
      {_outputs, vjps} = Mlx.Transforms.vjp(
        fn [x] -> [Nx.multiply(x, 2)] end,
        [Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)],
        [Nx.tensor([1.0, 1.0, 1.0], backend: Mlx.Backend)]
      )

      [vjp] = vjps
      assert Nx.to_flat_list(vjp) == [2.0, 2.0, 2.0]
    end
  end

  describe "Mlx.Transforms.jvp" do
    test "Jacobian-vector product of identity" do
      # f(x) = x, J = I, jvp = I * tangent = tangent
      {_outputs, jvps} = Mlx.Transforms.jvp(
        fn [x] -> [x] end,
        [Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)],
        [Nx.tensor([1.0, 1.0, 1.0], backend: Mlx.Backend)]
      )

      [jvp] = jvps
      assert Nx.to_flat_list(jvp) == [1.0, 1.0, 1.0]
    end

    test "jvp of scalar multiply" do
      # f(x) = 3*x, J = 3I, jvp = 3 * tangent
      {_outputs, jvps} = Mlx.Transforms.jvp(
        fn [x] -> [Nx.multiply(x, 3)] end,
        [Nx.tensor([1.0, 2.0], backend: Mlx.Backend)],
        [Nx.tensor([1.0, 1.0], backend: Mlx.Backend)]
      )

      [jvp] = jvps
      assert Nx.to_flat_list(jvp) == [3.0, 3.0]
    end
  end

  describe "Mlx.Transforms.vmap" do
    test "element-wise square over batch" do
      # vmap f(x) = x^2 over a 2x3 matrix → applies per row
      vmap_fn = Mlx.Transforms.vmap(fn [x] -> [Nx.pow(x, 2)] end)

      input = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: Mlx.Backend)
      [result] = vmap_fn.([input])

      expected = [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]
      for {row_actual, row_exp} <- Enum.zip(Nx.to_list(result), expected) do
        for {actual, exp} <- Enum.zip(row_actual, row_exp) do
          assert_in_delta actual, exp, 1.0e-5
        end
      end
    end

    test "sum over batch" do
      # vmap f(x) = sum(x) over rows
      vmap_fn = Mlx.Transforms.vmap(fn [x] -> [Nx.sum(x)] end)

      input = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: Mlx.Backend)
      [result] = vmap_fn.([input])

      expected = [6.0, 15.0]
      for {actual, exp} <- Enum.zip(Nx.to_flat_list(result), expected) do
        assert_in_delta actual, exp, 1.0e-5
      end
    end

    test "multi-input vmap: element-wise multiply" do
      # vmap f(x, y) = x * y over batch
      vmap_fn = Mlx.Transforms.vmap(fn [x, y] -> [Nx.multiply(x, y)] end, [0, 0])

      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Mlx.Backend)
      y = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], backend: Mlx.Backend)
      [result] = vmap_fn.([x, y])

      expected = [[5.0, 12.0], [21.0, 32.0]]
      for {row_actual, row_exp} <- Enum.zip(Nx.to_list(result), expected) do
        for {actual, exp} <- Enum.zip(row_actual, row_exp) do
          assert_in_delta actual, exp, 1.0e-5
        end
      end
    end

    test "batched dot product" do
      # vmap f(x) = dot(x, w) over batch, where w is not vmapped
      # Use in_axes: [0, -1] where -1 means "don't vmap this arg"
      # But simpler: each row dotted with itself
      vmap_fn = Mlx.Transforms.vmap(fn [x] -> [Nx.sum(Nx.multiply(x, x))] end)

      input = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: Mlx.Backend)
      [result] = vmap_fn.([input])

      # [1+4+9, 16+25+36] = [14, 77]
      expected = [14.0, 77.0]
      for {actual, exp} <- Enum.zip(Nx.to_flat_list(result), expected) do
        assert_in_delta actual, exp, 1.0e-5
      end
    end

    test "vmap with add scalar" do
      # vmap f(x) = x + 1 over batch
      vmap_fn = Mlx.Transforms.vmap(fn [x] -> [Nx.add(x, 1)] end)

      input = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Mlx.Backend)
      [result] = vmap_fn.([input])

      expected = [[2.0, 3.0], [4.0, 5.0]]
      for {row_actual, row_exp} <- Enum.zip(Nx.to_list(result), expected) do
        for {actual, exp} <- Enum.zip(row_actual, row_exp) do
          assert_in_delta actual, exp, 1.0e-5
        end
      end
    end
  end

  describe "Mlx.Transforms with BinaryBackend inputs" do
    test "auto-transfers from BinaryBackend" do
      grad_fn = Mlx.Transforms.grad(fn [x] ->
        [Nx.sum(Nx.pow(x, 2))]
      end)

      # Inputs on BinaryBackend should be auto-transferred
      [grad] = grad_fn.([Nx.tensor([1.0, 2.0], backend: Nx.BinaryBackend)])

      expected = [2.0, 4.0]
      for {actual, exp} <- Enum.zip(Nx.to_flat_list(grad), expected) do
        assert_in_delta actual, exp, 1.0e-5
      end
    end
  end
end
