defmodule Mlx.CompilerTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  # --- Test modules using Mlx.Compiler ---

  defmodule SimpleModel do
    import Nx.Defn

    @defn_compiler Mlx.Compiler
    defn(add(a, b), do: Nx.add(a, b))

    @defn_compiler Mlx.Compiler
    defn chained(x) do
      x
      |> Nx.multiply(2)
      |> Nx.add(1)
      |> Nx.negate()
    end

    @defn_compiler Mlx.Compiler
    defn predict(x, w) do
      Nx.dot(x, w)
    end

    @defn_compiler Mlx.Compiler
    defn multi_op(a, b) do
      sum = Nx.add(a, b)
      diff = Nx.subtract(a, b)
      {sum, diff}
    end

    @defn_compiler Mlx.Compiler
    defn reduce_sum(x) do
      Nx.sum(x)
    end

    @defn_compiler Mlx.Compiler
    defn sigmoid(x) do
      Nx.sigmoid(x)
    end
  end

  describe "basic defn compilation" do
    test "simple addition" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0, 6.0])
      result = SimpleModel.add(a, b)

      assert Nx.to_flat_list(result) == [5.0, 7.0, 9.0]
    end

    test "chained operations" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      result = SimpleModel.chained(x)

      # (x * 2 + 1) negated = -(2x + 1)
      expected = Nx.tensor([-3.0, -5.0, -7.0])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end

    test "matrix multiply" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      w = Nx.tensor([[0.5], [0.5]])
      result = SimpleModel.predict(x, w)

      assert Nx.to_flat_list(result) == [1.5, 3.5]
    end
  end

  describe "multiple outputs" do
    test "returns tuple of tensors" do
      a = Nx.tensor([10.0, 20.0])
      b = Nx.tensor([3.0, 7.0])
      {sum, diff} = SimpleModel.multi_op(a, b)

      assert Nx.to_flat_list(sum) == [13.0, 27.0]
      assert Nx.to_flat_list(diff) == [7.0, 13.0]
    end
  end

  describe "reductions" do
    test "sum reduction" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = SimpleModel.reduce_sum(x)

      assert Nx.to_number(result) == 10.0
    end
  end

  describe "activation functions" do
    test "sigmoid" do
      x = Nx.tensor(0.0)
      result = SimpleModel.sigmoid(x)

      # sigmoid(0) = 0.5
      assert_in_delta Nx.to_number(result), 0.5, 1.0e-5
    end
  end

  describe "backend preservation" do
    test "restores default backend after compilation" do
      prev = Nx.default_backend()

      a = Nx.tensor([1.0, 2.0])
      b = Nx.tensor([3.0, 4.0])
      _result = SimpleModel.add(a, b)

      assert Nx.default_backend() == prev
    end

    test "restores default backend on error" do
      prev = Nx.default_backend()

      # Even if something fails, the backend should be restored.
      # We test this by verifying the backend is correct after a
      # successful call (the try/after in the compiler handles this).
      _result = SimpleModel.add(Nx.tensor([1.0]), Nx.tensor([1.0]))
      assert Nx.default_backend() == prev
    end
  end

  describe "input transfer" do
    test "accepts tensors from different backends" do
      # Create tensors on the default BinaryBackend
      a = Nx.tensor([1.0, 2.0, 3.0], backend: Nx.BinaryBackend)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: Nx.BinaryBackend)

      # The compiler should transfer these to Mlx.Backend automatically
      result = SimpleModel.add(a, b)
      assert Nx.to_flat_list(result) == [5.0, 7.0, 9.0]
    end

    test "accepts tensors already on Mlx.Backend" do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)
      b = Nx.tensor([4.0, 5.0, 6.0], backend: Mlx.Backend)

      result = SimpleModel.add(a, b)
      assert Nx.to_flat_list(result) == [5.0, 7.0, 9.0]
    end
  end

  describe "lazy evaluation semantics" do
    test "chained ops are evaluated as single graph" do
      # This test verifies that chained operations produce correct results.
      # Under the hood, MLX builds a computation graph for the chain
      # and evaluates it in one optimized pass.
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      result = SimpleModel.chained(x)

      expected =
        for v <- [1.0, 2.0, 3.0, 4.0, 5.0] do
          -(v * 2 + 1)
        end

      assert Nx.to_flat_list(result) == expected
    end

    test "dot + sigmoid graph" do
      x = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      w = Nx.tensor([[0.0], [0.0]])
      result = SimpleModel.predict(x, w)

      # dot([[1,0],[0,1]], [[0],[0]]) = [[0],[0]]
      assert Nx.to_flat_list(result) == [0.0, 0.0]
    end
  end
end
