defmodule Mlx.ErrorQualityTest do
  @moduledoc """
  Tests that error messages are actionable and informative.

  These tests verify that when users make common mistakes (shape mismatches,
  invalid dtypes, out-of-bounds axes), the error messages contain enough
  context to diagnose and fix the issue.
  """
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  describe "shape mismatch errors" do
    test "matmul with incompatible dimensions mentions shapes" do
      a = Nx.tensor([[1.0, 2.0]])
      b = Nx.tensor([[1.0, 2.0]])

      error =
        assert_raise ArgumentError, fn ->
          # 1x2 @ 1x2 is invalid (need 1x2 @ 2xN)
          Nx.dot(a, b)
        end

      # Error should mention shapes or dimensions
      assert error.message =~ ~r/shape|dimension|dot|equal|compatible/i
    end

    test "add with incompatible broadcast shapes" do
      # Create tensors with shapes that can't broadcast: {2,3} + {4}
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      error =
        assert_raise ArgumentError, fn ->
          Nx.add(a, b)
        end

      # Nx catches broadcast errors before reaching MLX
      assert error.message =~ ~r/broadcast|shape|cannot|incompatible/i
    end
  end

  describe "dtype errors" do
    test "f64 tensor creation gives meaningful error" do
      error =
        assert_raise ArgumentError, fn ->
          # f64 is not supported by Metal
          Nx.tensor([1.0], type: :f64, backend: Mlx.Backend)
        end

      assert error.message =~ ~r/type|dtype|f64|unsupported|64/i
    end
  end

  describe "axis/dimension errors" do
    test "reduce with out-of-bounds axis mentions axis" do
      t = Nx.tensor([1.0, 2.0, 3.0])

      error =
        assert_raise ArgumentError, fn ->
          # axis 5 is out of bounds for a 1D tensor
          Nx.sum(t, axes: [5])
        end

      assert error.message =~ ~r/axis|axes|out of|rank|bound|dimension/i
    end

    test "reshape to incompatible size mentions shape" do
      t = Nx.tensor([1.0, 2.0, 3.0])

      error =
        assert_raise ArgumentError, fn ->
          # Can't reshape 3 elements to {2, 2}
          Nx.reshape(t, {2, 2})
        end

      assert error.message =~ ~r/shape|size|reshape|cannot/i
    end
  end

  describe "I/O errors" do
    test "load from non-existent file mentions file path" do
      error =
        assert_raise RuntimeError, fn ->
          Mlx.IO.load("/tmp/definitely_nonexistent_file_#{System.unique_integer()}.npy")
        end

      assert error.message =~ ~r/file|path|load|exist|found|open|MLX/i
    end
  end

  describe "type constraint errors" do
    test "bitwise ops on float tensors raise informative error" do
      # MLX can handle this at C level, but Nx validates first
      a = Nx.tensor([1.0, 2.0])

      error =
        assert_raise ArgumentError, fn ->
          Nx.bitwise_and(a, a)
        end

      # Should mention type requirement
      assert error.message =~ ~r/type|integer|bitwise|float|expected/i
    end
  end

  describe "pointer operation errors" do
    test "from_pointer raises with helpful message" do
      error =
        assert_raise ArgumentError, fn ->
          Mlx.Backend.from_pointer(nil, nil, nil, nil, nil)
        end

      assert error.message =~ "from_pointer"
      assert error.message =~ "Nx.from_binary"
    end

    test "to_pointer raises with helpful message" do
      t = Nx.tensor([1.0])

      error =
        assert_raise ArgumentError, fn ->
          Mlx.Backend.to_pointer(t, [])
        end

      assert error.message =~ "to_pointer"
      assert error.message =~ "Nx.to_binary"
    end
  end

  describe "transform errors" do
    test "value_and_grad with non-scalar output describes issue" do
      # value_and_grad requires scalar (0-dim) output for gradient
      vag_fn =
        Mlx.Transforms.value_and_grad(fn [x] ->
          # Return a vector, not a scalar — this should fail
          [x]
        end)

      error =
        assert_raise RuntimeError, fn ->
          vag_fn.([Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)])
        end

      # MLX should mention the gradient/scalar requirement
      assert error.message =~ ~r/grad|scalar|dimension|shape|MLX/i
    end
  end

  describe "slice errors" do
    test "slice with negative length gives informative error" do
      t = Nx.tensor([1.0, 2.0, 3.0])

      error =
        assert_raise ArgumentError, fn ->
          # Length must be positive
          Nx.slice(t, [0], [-1])
        end

      assert error.message =~ ~r/length|size|slice|positive|non.negative/i
    end
  end
end
