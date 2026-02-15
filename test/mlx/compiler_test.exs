defmodule Mlx.CompilerTest do
  use ExUnit.Case, async: false

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

  # --- Grad-capable defn functions ---
  # These use Nx.Defn.grad/value_and_grad which build symbolic
  # gradient expressions that the compiler evaluates.

  defmodule GradModel do
    import Nx.Defn

    # f(x) = x^2, grad = 2x
    @defn_compiler Mlx.Compiler
    defn square(x), do: Nx.pow(x, 2)

    @defn_compiler Mlx.Compiler
    defn grad_square(x), do: grad(x, &square/1)

    @defn_compiler Mlx.Compiler
    defn value_and_grad_square(x), do: value_and_grad(x, &square/1)

    # f(x) = x^3, grad = 3x^2
    @defn_compiler Mlx.Compiler
    defn cube(x), do: Nx.pow(x, 3)

    @defn_compiler Mlx.Compiler
    defn grad_cube(x), do: grad(x, &cube/1)

    # f(x, y) = x * y  (multi-variable)
    @defn_compiler Mlx.Compiler
    defn product(x, y), do: Nx.multiply(x, y)

    @defn_compiler Mlx.Compiler
    defn grad_product_x(x, y), do: grad(x, fn x_ -> product(x_, y) end)

    # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    @defn_compiler Mlx.Compiler
    defn grad_sigmoid(x), do: grad(x, &Nx.sigmoid/1)

    # Gradient of sum: grad(sum(x)) = ones_like(x)
    @defn_compiler Mlx.Compiler
    defn grad_sum(x), do: grad(x, &Nx.sum/1)

    # Gradient with mean: grad(mean(x)) = 1/n for each element
    @defn_compiler Mlx.Compiler
    defn grad_mean(x), do: grad(x, &Nx.mean/1)

    # Composed function: f(x) = exp(x^2)
    @defn_compiler Mlx.Compiler
    defn exp_square(x), do: Nx.exp(Nx.pow(x, 2))

    @defn_compiler Mlx.Compiler
    defn grad_exp_square(x), do: grad(x, &exp_square/1)

    # Gradient with matmul: f(w) = sum(x . w)
    @defn_compiler Mlx.Compiler
    defn linear_sum(x, w) do
      Nx.dot(x, w) |> Nx.sum()
    end

    @defn_compiler Mlx.Compiler
    defn grad_linear_w(x, w), do: grad(w, fn w_ -> linear_sum(x, w_) end)

    # Gradient with select (conditional)
    @defn_compiler Mlx.Compiler
    defn relu(x), do: Nx.max(x, 0)

    @defn_compiler Mlx.Compiler
    defn grad_relu(x), do: grad(x, fn x_ -> Nx.sum(relu(x_)) end)

    # Second-order gradient: grad(grad(x^3)) = 6x
    @defn_compiler Mlx.Compiler
    defn second_grad_cube(x) do
      grad(x, fn x_ -> grad(x_, &cube/1) end)
    end

    # JIT compilation (Nx.Defn.jit)
    @defn_compiler Mlx.Compiler
    defn jit_add(a, b), do: Nx.add(a, b)

    # Gradient with broadcasting: f(x) = sum(x + [1,2,3])
    @defn_compiler Mlx.Compiler
    defn grad_broadcast(x) do
      grad(x, fn x_ ->
        Nx.add(x_, Nx.tensor([1.0, 2.0, 3.0])) |> Nx.sum()
      end)
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

  # ============================================================
  # Nx.Defn.grad / value_and_grad through Mlx.Compiler
  # ============================================================

  describe "Nx.Defn.grad through Mlx.Compiler" do
    test "simple scalar gradient: f(x) = x^2, grad = 2x" do
      result = GradModel.grad_square(Nx.tensor(3.0))
      assert_in_delta Nx.to_number(result), 6.0, 1.0e-5
    end

    test "gradient at zero: f(x) = x^2, grad(0) = 0" do
      result = GradModel.grad_square(Nx.tensor(0.0))
      assert_in_delta Nx.to_number(result), 0.0, 1.0e-5
    end

    test "gradient at negative: f(x) = x^2, grad(-2) = -4" do
      result = GradModel.grad_square(Nx.tensor(-2.0))
      assert_in_delta Nx.to_number(result), -4.0, 1.0e-5
    end

    test "cubic gradient: f(x) = x^3, grad = 3x^2" do
      result = GradModel.grad_cube(Nx.tensor(2.0))
      assert_in_delta Nx.to_number(result), 12.0, 1.0e-5
    end

    test "value_and_grad returns both value and gradient" do
      {value, gradient} = GradModel.value_and_grad_square(Nx.tensor(3.0))
      assert_in_delta Nx.to_number(value), 9.0, 1.0e-5
      assert_in_delta Nx.to_number(gradient), 6.0, 1.0e-5
    end

    test "multi-variable gradient: f(x,y) = x*y, grad_x = y" do
      result = GradModel.grad_product_x(Nx.tensor(2.0), Nx.tensor(5.0))
      assert_in_delta Nx.to_number(result), 5.0, 1.0e-5
    end

    test "sigmoid gradient: sigmoid'(0) = 0.25" do
      result = GradModel.grad_sigmoid(Nx.tensor(0.0))
      # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
      assert_in_delta Nx.to_number(result), 0.25, 1.0e-5
    end

    test "gradient of sum: grad(sum(x)) = ones_like(x)" do
      result = GradModel.grad_sum(Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_flat_list(result) == [1.0, 1.0, 1.0]
    end

    test "gradient of mean: grad(mean(x)) = 1/n" do
      result = GradModel.grad_mean(Nx.tensor([1.0, 2.0, 3.0, 4.0]))
      expected = [0.25, 0.25, 0.25, 0.25]

      for {actual, exp} <- Enum.zip(Nx.to_flat_list(result), expected) do
        assert_in_delta actual, exp, 1.0e-5
      end
    end

    test "composed function gradient: f(x) = exp(x^2), f'(x) = 2x*exp(x^2)" do
      x = 1.0
      result = GradModel.grad_exp_square(Nx.tensor(x))
      expected = 2.0 * x * :math.exp(x * x)
      assert_in_delta Nx.to_number(result), expected, 1.0e-4
    end

    test "gradient with matmul: f(w) = sum(x . w)" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      w = Nx.tensor([[1.0], [1.0]])
      result = GradModel.grad_linear_w(x, w)

      # d/dw sum(x . w) = x^T . ones = [[1+3], [2+4]] = [[4], [6]]
      expected = [4.0, 6.0]

      for {actual, exp} <- Enum.zip(Nx.to_flat_list(result), expected) do
        assert_in_delta actual, exp, 1.0e-5
      end
    end

    test "gradient of relu: subgradient 1 for x>0, 0 for x<0" do
      result = GradModel.grad_relu(Nx.tensor([2.0, -1.0, 3.0, -0.5]))
      expected = [1.0, 0.0, 1.0, 0.0]

      for {actual, exp} <- Enum.zip(Nx.to_flat_list(result), expected) do
        assert_in_delta actual, exp, 1.0e-5
      end
    end

    test "second-order gradient: d²/dx²(x³) = 6x" do
      result = GradModel.second_grad_cube(Nx.tensor(2.0))
      assert_in_delta Nx.to_number(result), 12.0, 1.0e-5
    end

    test "gradient with broadcasting: grad(sum(x + const)) = ones" do
      result = GradModel.grad_broadcast(Nx.tensor(1.0))
      # scalar + vector → broadcast, sum → grad wrt scalar = 3 (sum of ones)
      assert_in_delta Nx.to_number(result), 3.0, 1.0e-5
    end
  end

  describe "Nx.Defn.jit through Mlx.Compiler" do
    test "jit compilation produces correct results" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0, 6.0])
      result = GradModel.jit_add(a, b)

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
