defmodule MlxTest do
  use ExUnit.Case
  # Validates: U1 (Nx.tensor works), RT-1 (array lifecycle), RT-5 (Nx.Backend)

  @moduletag :integration

  setup do
    # Ensure MLX backend is available
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  describe "tensor creation" do
    test "creates tensor from list" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      assert Nx.shape(t) == {3}
      assert Nx.type(t) == {:f, 32}
    end

    test "creates 2D tensor" do
      t = Nx.tensor([[1, 2], [3, 4]])
      assert Nx.shape(t) == {2, 2}
    end

    test "creates tensor with explicit backend" do
      t = Nx.tensor([1, 2, 3], backend: Mlx.Backend)
      assert %Mlx.Backend{} = t.data
    end

    test "creates tensor with explicit type" do
      t = Nx.tensor([1, 2, 3], type: {:f, 32})
      assert Nx.type(t) == {:f, 32}
    end
  end

  describe "tensor data roundtrip" do
    test "from_binary -> to_binary preserves data" do
      data = <<1.0::float-native-32, 2.0::float-native-32, 3.0::float-native-32>>
      t = Nx.from_binary(data, {:f, 32})
      assert Nx.to_binary(t) == data
    end

    test "integer data roundtrip" do
      t = Nx.tensor([1, 2, 3, 4], type: {:s, 32})
      assert Nx.to_flat_list(t) == [1, 2, 3, 4]
    end
  end

  describe "basic arithmetic" do
    test "addition" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0, 6.0])
      result = Nx.add(a, b)
      assert Nx.to_flat_list(result) == [5.0, 7.0, 9.0]
    end

    test "subtraction" do
      result = Nx.subtract(Nx.tensor([5.0, 3.0]), Nx.tensor([1.0, 1.0]))
      assert Nx.to_flat_list(result) == [4.0, 2.0]
    end

    test "multiplication" do
      result = Nx.multiply(Nx.tensor([2.0, 3.0]), Nx.tensor([4.0, 5.0]))
      assert Nx.to_flat_list(result) == [8.0, 15.0]
    end

    test "division" do
      result = Nx.divide(Nx.tensor([10.0, 6.0]), Nx.tensor([2.0, 3.0]))
      assert Nx.to_flat_list(result) == [5.0, 2.0]
    end

    test "negation" do
      result = Nx.negate(Nx.tensor([1.0, -2.0, 3.0]))
      assert Nx.to_flat_list(result) == [-1.0, 2.0, -3.0]
    end
  end

  describe "math functions" do
    test "exp" do
      result = Nx.exp(Nx.tensor([0.0, 1.0]))
      [a, b] = Nx.to_flat_list(result)
      assert_in_delta a, 1.0, 1.0e-5
      assert_in_delta b, :math.exp(1), 1.0e-5
    end

    test "log" do
      result = Nx.log(Nx.tensor([1.0, :math.exp(1)]))
      [a, b] = Nx.to_flat_list(result)
      assert_in_delta a, 0.0, 1.0e-5
      assert_in_delta b, 1.0, 1.0e-5
    end

    test "sqrt" do
      result = Nx.sqrt(Nx.tensor([4.0, 9.0, 16.0]))
      assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0]
    end
  end

  describe "reductions" do
    test "sum" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      assert Nx.to_number(Nx.sum(t)) == 10.0
    end

    test "sum with axis" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = Nx.sum(t, axes: [0])
      assert Nx.to_flat_list(result) == [4.0, 6.0]
    end

    test "mean" do
      t = Nx.tensor([2.0, 4.0, 6.0])
      assert Nx.to_number(Nx.mean(t)) == 4.0
    end

    test "max" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
      assert Nx.to_number(Nx.reduce_max(t)) == 5.0
    end
  end

  describe "shape operations" do
    test "reshape" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      result = Nx.reshape(t, {2, 3})
      assert Nx.shape(result) == {2, 3}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    test "transpose" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      result = Nx.transpose(t)
      assert Nx.shape(result) == {3, 2}
    end

    test "broadcast" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.broadcast(t, {2, 3})
      assert Nx.shape(result) == {2, 3}
    end
  end

  describe "comparison" do
    test "equal" do
      result = Nx.equal(Nx.tensor([1, 2, 3]), Nx.tensor([1, 0, 3]))
      assert Nx.to_flat_list(result) == [1, 0, 1]
    end

    test "less" do
      result = Nx.less(Nx.tensor([1, 2, 3]), Nx.tensor([2, 2, 2]))
      assert Nx.to_flat_list(result) == [1, 0, 0]
    end
  end

  describe "linear algebra" do
    test "matmul via dot" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
      result = Nx.dot(a, b)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [19.0, 22.0, 43.0, 50.0]
    end
  end

  describe "eye" do
    test "creates identity matrix" do
      result = Nx.eye(3)
      assert Nx.shape(result) == {3, 3}
      assert Nx.to_flat_list(result) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    end
  end

  describe "type conversion" do
    test "as_type" do
      t = Nx.tensor([1, 2, 3], type: {:s, 32})
      result = Nx.as_type(t, {:f, 32})
      assert Nx.type(result) == {:f, 32}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end
  end

  describe "select" do
    test "conditional selection" do
      pred = Nx.tensor([1, 0, 1], type: {:u, 8})
      on_true = Nx.tensor([10.0, 20.0, 30.0])
      on_false = Nx.tensor([100.0, 200.0, 300.0])
      result = Nx.select(pred, on_true, on_false)
      assert Nx.to_flat_list(result) == [10.0, 200.0, 30.0]
    end
  end

  describe "clip" do
    test "clips values to range" do
      t = Nx.tensor([1.0, 5.0, 10.0, 15.0, 20.0])
      result = Nx.clip(t, Nx.tensor(5.0), Nx.tensor(15.0))
      assert Nx.to_flat_list(result) == [5.0, 5.0, 10.0, 15.0, 15.0]
    end

    test "clip with integer tensors" do
      t = Nx.tensor([0, 3, 7, 10], type: {:s, 32})
      result = Nx.clip(t, Nx.tensor(2, type: {:s, 32}), Nx.tensor(8, type: {:s, 32}))
      assert Nx.to_flat_list(result) == [2, 3, 7, 8]
    end
  end

  describe "sort" do
    test "sort ascending" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
      result = Nx.sort(t)
      assert Nx.to_flat_list(result) == [1.0, 1.0, 3.0, 4.0, 5.0]
    end

    test "sort descending" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
      result = Nx.sort(t, direction: :desc)
      assert Nx.to_flat_list(result) == [5.0, 4.0, 3.0, 1.0, 1.0]
    end

    test "sort along axis" do
      t = Nx.tensor([[3.0, 1.0], [4.0, 2.0]])
      result = Nx.sort(t, axis: 1)
      assert Nx.to_flat_list(result) == [1.0, 3.0, 2.0, 4.0]
    end
  end

  describe "argsort" do
    test "argsort ascending" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
      result = Nx.argsort(t)
      indices = Nx.to_flat_list(result)
      # Indices 1,3 (value 1.0), then 0 (3.0), 2 (4.0), 4 (5.0)
      assert Enum.at(indices, 0) in [1, 3]
      assert Enum.at(indices, 1) in [1, 3]
      assert Enum.at(indices, 2) == 0
      assert Enum.at(indices, 3) == 2
      assert Enum.at(indices, 4) == 4
    end
  end

  describe "bitwise operations" do
    test "bitwise_and" do
      a = Nx.tensor([0b1100, 0b1010], type: {:u, 8})
      b = Nx.tensor([0b1010, 0b1100], type: {:u, 8})
      result = Nx.bitwise_and(a, b)
      assert Nx.to_flat_list(result) == [0b1000, 0b1000]
    end

    test "bitwise_or" do
      a = Nx.tensor([0b1100, 0b1010], type: {:u, 8})
      b = Nx.tensor([0b1010, 0b1100], type: {:u, 8})
      result = Nx.bitwise_or(a, b)
      assert Nx.to_flat_list(result) == [0b1110, 0b1110]
    end

    test "bitwise_xor" do
      a = Nx.tensor([0b1100, 0b1010], type: {:u, 8})
      b = Nx.tensor([0b1010, 0b1100], type: {:u, 8})
      result = Nx.bitwise_xor(a, b)
      assert Nx.to_flat_list(result) == [0b0110, 0b0110]
    end

    test "left_shift" do
      a = Nx.tensor([1, 2, 3], type: {:u, 32})
      b = Nx.tensor([1, 2, 3], type: {:u, 32})
      result = Nx.left_shift(a, b)
      assert Nx.to_flat_list(result) == [2, 8, 24]
    end

    test "right_shift" do
      a = Nx.tensor([8, 16, 32], type: {:u, 32})
      b = Nx.tensor([1, 2, 3], type: {:u, 32})
      result = Nx.right_shift(a, b)
      assert Nx.to_flat_list(result) == [4, 4, 4]
    end
  end

  describe "reverse" do
    test "reverse 1D tensor" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = Nx.reverse(t)
      assert Nx.to_flat_list(result) == [4.0, 3.0, 2.0, 1.0]
    end

    test "reverse 2D along axis 0" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = Nx.reverse(t, axes: [0])
      assert Nx.to_flat_list(result) == [3.0, 4.0, 1.0, 2.0]
    end

    test "reverse 2D along axis 1" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = Nx.reverse(t, axes: [1])
      assert Nx.to_flat_list(result) == [2.0, 1.0, 4.0, 3.0]
    end
  end

  describe "concatenate" do
    test "concatenate along axis 0" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[5.0, 6.0]])
      result = Nx.concatenate([a, b], axis: 0)
      assert Nx.shape(result) == {3, 2}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    test "concatenate along axis 1" do
      a = Nx.tensor([[1.0], [2.0]])
      b = Nx.tensor([[3.0, 4.0], [5.0, 6.0]])
      result = Nx.concatenate([a, b], axis: 1)
      assert Nx.shape(result) == {2, 3}
      assert Nx.to_flat_list(result) == [1.0, 3.0, 4.0, 2.0, 5.0, 6.0]
    end

    test "concatenate multiple tensors" do
      a = Nx.tensor([1.0, 2.0])
      b = Nx.tensor([3.0])
      c = Nx.tensor([4.0, 5.0, 6.0])
      result = Nx.concatenate([a, b, c])
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end
  end

  describe "take" do
    test "take elements by index" do
      t = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
      indices = Nx.tensor([0, 2, 4], type: {:s, 32})
      result = Nx.take(t, indices, axis: 0)
      assert Nx.to_flat_list(result) == [10.0, 30.0, 50.0]
    end

    test "take rows from 2D" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      indices = Nx.tensor([0, 2], type: {:s, 32})
      result = Nx.take(t, indices, axis: 0)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 5.0, 6.0]
    end
  end

  describe "pad" do
    test "pad 1D tensor" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.pad(t, 0.0, [{1, 2, 0}])
      assert Nx.shape(result) == {6}
      assert Nx.to_flat_list(result) == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
    end

    test "pad 2D tensor" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = Nx.pad(t, 0.0, [{1, 0, 0}, {0, 1, 0}])
      assert Nx.shape(result) == {3, 3}
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0]
    end
  end

  describe "additional unary ops" do
    test "abs" do
      t = Nx.tensor([-1.0, 0.0, 1.0])
      assert Nx.to_flat_list(Nx.abs(t)) == [1.0, 0.0, 1.0]
    end

    test "sign" do
      t = Nx.tensor([-5.0, 0.0, 3.0])
      assert Nx.to_flat_list(Nx.sign(t)) == [-1.0, 0.0, 1.0]
    end

    test "ceil and floor" do
      t = Nx.tensor([1.3, 2.7, -0.5])
      assert Nx.to_flat_list(Nx.ceil(t)) == [2.0, 3.0, 0.0]
      assert Nx.to_flat_list(Nx.floor(t)) == [1.0, 2.0, -1.0]
    end

    test "rsqrt" do
      t = Nx.tensor([4.0, 9.0])
      [a, b] = Nx.to_flat_list(Nx.rsqrt(t))
      assert_in_delta a, 0.5, 1.0e-5
      assert_in_delta b, 1.0 / 3.0, 1.0e-5
    end

    test "erfc" do
      t = Nx.tensor(0.0)
      result = Nx.to_number(Nx.erfc(t))
      assert_in_delta result, 1.0, 1.0e-5
    end

    test "trig functions" do
      t = Nx.tensor(0.0)
      assert_in_delta Nx.to_number(Nx.sin(t)), 0.0, 1.0e-5
      assert_in_delta Nx.to_number(Nx.cos(t)), 1.0, 1.0e-5
      assert_in_delta Nx.to_number(Nx.tan(t)), 0.0, 1.0e-5
    end

    test "hyperbolic functions" do
      t = Nx.tensor(0.0)
      assert_in_delta Nx.to_number(Nx.sinh(t)), 0.0, 1.0e-5
      assert_in_delta Nx.to_number(Nx.cosh(t)), 1.0, 1.0e-5
      assert_in_delta Nx.to_number(Nx.tanh(t)), 0.0, 1.0e-5
    end

    test "logical_not" do
      t = Nx.tensor([1, 0, 1], type: {:u, 8})
      assert Nx.to_flat_list(Nx.logical_not(t)) == [0, 1, 0]
    end
  end

  describe "additional binary ops" do
    test "remainder" do
      a = Nx.tensor([7.0, 10.0, 13.0])
      b = Nx.tensor([3.0, 4.0, 5.0])
      result = Nx.remainder(a, b)
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end

    test "power" do
      a = Nx.tensor([2.0, 3.0, 4.0])
      b = Nx.tensor([3.0, 2.0, 0.5])
      result = Nx.pow(a, b)
      [x, y, z] = Nx.to_flat_list(result)
      assert_in_delta x, 8.0, 1.0e-5
      assert_in_delta y, 9.0, 1.0e-5
      assert_in_delta z, 2.0, 1.0e-5
    end

    test "quotient" do
      a = Nx.tensor([7, 10, 15], type: {:s, 32})
      b = Nx.tensor([2, 3, 4], type: {:s, 32})
      result = Nx.quotient(a, b)
      assert Nx.to_flat_list(result) == [3, 3, 3]
    end

    test "logical_and" do
      a = Nx.tensor([1, 0, 1], type: {:u, 8})
      b = Nx.tensor([1, 1, 0], type: {:u, 8})
      result = Nx.logical_and(a, b)
      assert Nx.to_flat_list(result) == [1, 0, 0]
    end

    test "logical_or" do
      a = Nx.tensor([1, 0, 0], type: {:u, 8})
      b = Nx.tensor([0, 0, 1], type: {:u, 8})
      result = Nx.logical_or(a, b)
      assert Nx.to_flat_list(result) == [1, 0, 1]
    end

    test "not_equal" do
      a = Nx.tensor([1, 2, 3])
      b = Nx.tensor([1, 0, 3])
      result = Nx.not_equal(a, b)
      assert Nx.to_flat_list(result) == [0, 1, 0]
    end

    test "greater" do
      a = Nx.tensor([1, 2, 3])
      b = Nx.tensor([2, 2, 2])
      result = Nx.greater(a, b)
      assert Nx.to_flat_list(result) == [0, 0, 1]
    end
  end

  describe "additional reductions" do
    test "product" do
      t = Nx.tensor([2.0, 3.0, 4.0])
      assert Nx.to_number(Nx.product(t)) == 24.0
    end

    test "min reduction" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
      assert Nx.to_number(Nx.reduce_min(t)) == 1.0
    end

    test "argmax" do
      t = Nx.tensor([1.0, 5.0, 3.0])
      assert Nx.to_number(Nx.argmax(t)) == 1
    end

    test "argmin" do
      t = Nx.tensor([3.0, 1.0, 5.0])
      assert Nx.to_number(Nx.argmin(t)) == 1
    end

    test "sum with keep_axes" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = Nx.sum(t, axes: [1], keep_axes: true)
      assert Nx.shape(result) == {2, 1}
      assert Nx.to_flat_list(result) == [3.0, 7.0]
    end
  end

  describe "additional shape ops" do
    test "squeeze" do
      t = Nx.tensor([[[1.0, 2.0, 3.0]]])
      result = Nx.squeeze(t)
      assert Nx.shape(result) == {3}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end

    test "iota" do
      result = Nx.iota({2, 3}, type: {:f, 32})
      assert Nx.shape(result) == {2, 3}
      assert Nx.to_flat_list(result) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    end

    test "iota with axis" do
      result = Nx.iota({2, 3}, type: {:f, 32}, axis: 1)
      assert Nx.to_flat_list(result) == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    end

    test "slice" do
      t = Nx.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
      result = Nx.slice(t, [1], [3])
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end

    test "constant" do
      t = Nx.broadcast(3.14, {2, 3})
      assert Nx.shape(t) == {2, 3}
      values = Nx.to_flat_list(t)
      assert length(values) == 6
      Enum.each(values, fn v -> assert_in_delta v, 3.14, 1.0e-5 end)
    end
  end

  describe "element-wise max/min" do
    test "max selects element-wise maximum" do
      a = Nx.tensor([1.0, 5.0, 3.0])
      b = Nx.tensor([4.0, 2.0, 6.0])
      result = Nx.max(a, b)
      assert Nx.to_flat_list(result) == [4.0, 5.0, 6.0]
    end

    test "min selects element-wise minimum" do
      a = Nx.tensor([1.0, 5.0, 3.0])
      b = Nx.tensor([4.0, 2.0, 6.0])
      result = Nx.min(a, b)
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end

    test "max with 2D tensors" do
      a = Nx.tensor([[1.0, 6.0], [3.0, 4.0]])
      b = Nx.tensor([[5.0, 2.0], [7.0, 0.0]])
      result = Nx.max(a, b)
      assert Nx.to_flat_list(result) == [5.0, 6.0, 7.0, 4.0]
    end

    test "min with broadcasting" do
      a = Nx.tensor([[1.0, 5.0], [3.0, 2.0]])
      b = Nx.tensor([3.0, 3.0])
      result = Nx.min(a, b)
      assert Nx.to_flat_list(result) == [1.0, 3.0, 3.0, 2.0]
    end
  end

  describe "bitwise_not" do
    test "inverts bits" do
      t = Nx.tensor([0, 255], type: {:u, 8})
      result = Nx.bitwise_not(t)
      assert Nx.to_flat_list(result) == [255, 0]
    end

    test "inverts signed integer" do
      t = Nx.tensor([0, 1, -1], type: {:s, 32})
      result = Nx.bitwise_not(t)
      assert Nx.to_flat_list(result) == [-1, -2, 0]
    end
  end

  describe "logical_xor" do
    test "xor on boolean-like tensors" do
      a = Nx.tensor([1, 0, 1, 0], type: {:u, 8})
      b = Nx.tensor([1, 1, 0, 0], type: {:u, 8})
      result = Nx.logical_xor(a, b)
      assert Nx.to_flat_list(result) == [0, 1, 1, 0]
    end
  end

  describe "all/any reductions" do
    test "all returns true when all elements are true" do
      t = Nx.tensor([1, 1, 1], type: {:u, 8})
      assert Nx.to_number(Nx.all(t)) == 1
    end

    test "all returns false when any element is false" do
      t = Nx.tensor([1, 0, 1], type: {:u, 8})
      assert Nx.to_number(Nx.all(t)) == 0
    end

    test "any returns true when any element is true" do
      t = Nx.tensor([0, 0, 1], type: {:u, 8})
      assert Nx.to_number(Nx.any(t)) == 1
    end

    test "any returns false when all elements are false" do
      t = Nx.tensor([0, 0, 0], type: {:u, 8})
      assert Nx.to_number(Nx.any(t)) == 0
    end

    test "all with axis" do
      t = Nx.tensor([[1, 1], [1, 0]], type: {:u, 8})
      result = Nx.all(t, axes: [0])
      assert Nx.to_flat_list(result) == [1, 0]
    end

    test "any with axis" do
      t = Nx.tensor([[0, 0], [1, 0]], type: {:u, 8})
      result = Nx.any(t, axes: [0])
      assert Nx.to_flat_list(result) == [1, 0]
    end
  end

  describe "stack" do
    test "stack 1D tensors along axis 0" do
      a = Nx.tensor([1.0, 2.0])
      b = Nx.tensor([3.0, 4.0])
      result = Nx.stack([a, b])
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0, 4.0]
    end

    test "stack along axis 1" do
      a = Nx.tensor([1.0, 2.0])
      b = Nx.tensor([3.0, 4.0])
      result = Nx.stack([a, b], axis: 1)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [1.0, 3.0, 2.0, 4.0]
    end

    test "stack three tensors" do
      a = Nx.tensor([1.0, 2.0])
      b = Nx.tensor([3.0, 4.0])
      c = Nx.tensor([5.0, 6.0])
      result = Nx.stack([a, b, c])
      assert Nx.shape(result) == {3, 2}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end
  end

  describe "put_slice" do
    test "put_slice into 1D tensor" do
      t = Nx.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
      slice = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.put_slice(t, [1], slice)
      assert Nx.to_flat_list(result) == [0.0, 1.0, 2.0, 3.0, 0.0]
    end

    test "put_slice into 2D tensor" do
      t = Nx.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
      slice = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = Nx.put_slice(t, [0, 1], slice)
      assert Nx.to_flat_list(result) == [0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0]
    end
  end

  describe "bitcast" do
    test "bitcast f32 to u32" do
      t = Nx.tensor([1.0], type: {:f, 32})
      result = Nx.bitcast(t, {:u, 32})
      assert Nx.type(result) == {:u, 32}
      # IEEE 754: 1.0f = 0x3F800000 = 1065353216
      assert Nx.to_flat_list(result) == [1_065_353_216]
    end

    test "bitcast preserves shape" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
      result = Nx.bitcast(t, {:s, 32})
      assert Nx.shape(result) == {2, 2}
      assert Nx.type(result) == {:s, 32}
    end
  end

  describe "cbrt" do
    test "cube root of perfect cubes" do
      t = Nx.tensor([8.0, 27.0, 64.0])
      result = Nx.cbrt(t)
      [a, b, c] = Nx.to_flat_list(result)
      assert_in_delta a, 2.0, 1.0e-5
      assert_in_delta b, 3.0, 1.0e-5
      assert_in_delta c, 4.0, 1.0e-4
    end

    test "cube root of 1 is 1" do
      t = Nx.tensor(1.0)
      result = Nx.cbrt(t)
      assert_in_delta Nx.to_number(result), 1.0, 1.0e-6
    end
  end

  describe "numerical accuracy" do
    test "sigmoid approximation" do
      # sigmoid(x) = 1 / (1 + exp(-x))
      x = Nx.tensor([0.0, 1.0, -1.0])
      result = Nx.sigmoid(x)
      expected = [0.5, 0.7310586, 0.2689414]

      Enum.zip(Nx.to_flat_list(result), expected)
      |> Enum.each(fn {got, exp} -> assert_in_delta got, exp, 1.0e-4 end)
    end

    test "log-exp roundtrip" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.log(Nx.exp(x))

      Enum.zip(Nx.to_flat_list(result), [1.0, 2.0, 3.0])
      |> Enum.each(fn {got, exp} -> assert_in_delta got, exp, 1.0e-5 end)
    end

    test "sqrt-square roundtrip" do
      x = Nx.tensor([4.0, 9.0, 16.0])
      result = Nx.pow(Nx.sqrt(x), 2)

      Enum.zip(Nx.to_flat_list(result), [4.0, 9.0, 16.0])
      |> Enum.each(fn {got, exp} -> assert_in_delta got, exp, 1.0e-4 end)
    end

    test "trig identity sin^2 + cos^2 = 1" do
      x = Nx.tensor([0.0, 0.5, 1.0, 2.0])
      sin_sq = Nx.pow(Nx.sin(x), 2)
      cos_sq = Nx.pow(Nx.cos(x), 2)
      result = Nx.add(sin_sq, cos_sq)
      Enum.each(Nx.to_flat_list(result), fn v -> assert_in_delta v, 1.0, 1.0e-5 end)
    end

    test "matmul identity" do
      eye = Nx.eye(3)
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
      result = Nx.dot(x, eye)
      assert Nx.to_flat_list(result) == Nx.to_flat_list(x)
    end
  end
end
