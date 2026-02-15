defmodule Mlx.G11OpsTest do
  @moduledoc """
  Tests for the 8 ops from G11 that were previously unsupported:
  - count_leading_zeros (via BinaryBackend)
  - population_count (via BinaryBackend)
  - reduce (via BinaryBackend)
  - window_reduce (via BinaryBackend)
  - window_scatter_max (via BinaryBackend)
  - window_scatter_min (via BinaryBackend)
  - from_pointer (raises with helpful message)
  - to_pointer (raises with helpful message)
  """
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  # --- count_leading_zeros ---

  describe "count_leading_zeros" do
    test "u8 basic values" do
      # 0 -> 8, 1 -> 7, 2 -> 6, 4 -> 5, 128 -> 0, 255 -> 0
      t = Nx.tensor([0, 1, 2, 4, 128, 255], type: :u8)
      result = Nx.count_leading_zeros(t)
      assert Nx.to_flat_list(result) == [8, 7, 6, 5, 0, 0]
    end

    test "s32 basic values" do
      # 0 -> 32, 1 -> 31, 256 -> 23, -1 -> 0 (MSB is 1)
      t = Nx.tensor([0, 1, 256, -1], type: :s32)
      result = Nx.count_leading_zeros(t)
      assert Nx.to_flat_list(result) == [32, 31, 23, 0]
    end

    test "u16 powers of two" do
      t = Nx.tensor([1, 2, 4, 8, 16, 32, 64], type: :u16)
      result = Nx.count_leading_zeros(t)
      # 1=0b1 -> 15, 2=0b10 -> 14, 4=0b100 -> 13, ...
      assert Nx.to_flat_list(result) == [15, 14, 13, 12, 11, 10, 9]
    end

    test "preserves shape" do
      t = Nx.tensor([[0, 1], [2, 3]], type: :u8)
      result = Nx.count_leading_zeros(t)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [8, 7, 6, 6]
    end
  end

  # --- population_count ---

  describe "population_count" do
    test "u8 basic values" do
      # 0 -> 0, 1 -> 1, 3 -> 2, 7 -> 3, 255 -> 8
      t = Nx.tensor([0, 1, 3, 7, 255], type: :u8)
      result = Nx.population_count(t)
      assert Nx.to_flat_list(result) == [0, 1, 2, 3, 8]
    end

    test "s32 values" do
      # 0 -> 0, 1 -> 1, 15 -> 4, -1 -> 32 (all bits set)
      t = Nx.tensor([0, 1, 15, -1], type: :s32)
      result = Nx.population_count(t)
      assert Nx.to_flat_list(result) == [0, 1, 4, 32]
    end

    test "u16 specific patterns" do
      # 0xAAAA = 0b1010101010101010 -> 8 bits
      # 0x5555 = 0b0101010101010101 -> 8 bits
      t = Nx.tensor([0xAAAA, 0x5555, 0xFFFF], type: :u16)
      result = Nx.population_count(t)
      assert Nx.to_flat_list(result) == [8, 8, 16]
    end

    test "preserves shape" do
      t = Nx.tensor([[1, 3], [7, 15]], type: :u8)
      result = Nx.population_count(t)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [1, 2, 3, 4]
    end
  end

  # --- reduce ---

  describe "reduce" do
    test "sum via reduce" do
      t = Nx.tensor([1, 2, 3, 4, 5])
      result = Nx.reduce(t, 0, fn x, acc -> Nx.add(x, acc) end)
      assert Nx.to_number(result) == 15
    end

    test "product via reduce" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = Nx.reduce(t, 1, fn x, acc -> Nx.multiply(x, acc) end)
      assert_in_delta Nx.to_number(result), 24.0, 1.0e-5
    end

    test "max via reduce" do
      t = Nx.tensor([3, 1, 4, 1, 5, 9, 2, 6])
      result = Nx.reduce(t, Nx.Constants.min(:s32), fn x, acc ->
        Nx.select(Nx.greater(x, acc), x, acc)
      end)
      assert Nx.to_number(result) == 9
    end

    test "reduce with axis" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      result = Nx.reduce(t, 0, [axes: [1]], fn x, acc -> Nx.add(x, acc) end)
      assert Nx.to_flat_list(result) == [6, 15]
    end
  end

  # --- window_reduce ---

  describe "window_reduce" do
    test "custom sum via window_reduce" do
      t = Nx.tensor([1, 2, 3, 4, 5])
      result = Nx.window_reduce(t, 0, {3}, fn x, acc -> Nx.add(x, acc) end)
      assert Nx.to_flat_list(result) == [6, 9, 12]
    end

    test "custom max via window_reduce" do
      t = Nx.tensor([3, 1, 4, 1, 5])
      result = Nx.window_reduce(t, Nx.Constants.min(:s32), {3}, fn x, acc ->
        Nx.select(Nx.greater(x, acc), x, acc)
      end)
      assert Nx.to_flat_list(result) == [4, 4, 5]
    end

    test "2D window_reduce" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      result = Nx.window_reduce(t, 0, {2, 2}, fn x, acc -> Nx.add(x, acc) end)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [12, 16, 24, 28]
    end
  end

  # --- window_scatter_max ---

  describe "window_scatter_max" do
    test "1D basic" do
      t = Nx.tensor([1, 2, 3, 4, 5])
      source = Nx.tensor([10, 20, 30])
      init = Nx.tensor(0)
      result = Nx.window_scatter_max(t, source, init, {3})
      # Window 0: [1,2,3] → max at index 2 → scatter 10
      # Window 1: [2,3,4] → max at index 3 → scatter 20
      # Window 2: [3,4,5] → max at index 4 → scatter 30
      assert Nx.shape(result) == {5}
      assert Nx.to_flat_list(result) == [0, 0, 10, 20, 30]
    end

    test "2D basic" do
      t = Nx.tensor([[1, 2], [3, 4]])
      source = Nx.tensor([[10]])
      init = Nx.tensor(0)
      result = Nx.window_scatter_max(t, source, init, {2, 2})
      # Max is at position [1,1] (value 4)
      assert Nx.shape(result) == {2, 2}
      result_list = Nx.to_flat_list(result)
      assert Enum.at(result_list, 3) == 10
    end
  end

  # --- window_scatter_min ---

  describe "window_scatter_min" do
    test "1D basic" do
      t = Nx.tensor([5, 3, 4, 1, 2])
      source = Nx.tensor([10, 20, 30])
      init = Nx.tensor(0)
      result = Nx.window_scatter_min(t, source, init, {3})
      # Min in [5,3,4] at index 1, [3,4,1] at index 3, [4,1,2] at index 3
      assert Nx.shape(result) == {5}
      result_list = Nx.to_flat_list(result)
      assert Enum.at(result_list, 1) == 10
      # Index 3 receives contributions from windows where 1 is min
      assert Enum.at(result_list, 3) == 50
    end

    test "2D basic" do
      t = Nx.tensor([[4, 3], [2, 1]])
      source = Nx.tensor([[10]])
      init = Nx.tensor(0)
      result = Nx.window_scatter_min(t, source, init, {2, 2})
      # Min is at position [1,1] (value 1)
      assert Nx.shape(result) == {2, 2}
      result_list = Nx.to_flat_list(result)
      assert Enum.at(result_list, 3) == 10
    end
  end

  # --- from_pointer / to_pointer (should raise with helpful messages) ---

  describe "from_pointer" do
    test "raises with helpful message" do
      assert_raise ArgumentError, ~r/from_pointer is not supported.*Nx\.from_binary/, fn ->
        Mlx.Backend.from_pointer(nil, nil, nil, nil, nil)
      end
    end
  end

  describe "to_pointer" do
    test "raises with helpful message" do
      assert_raise ArgumentError, ~r/to_pointer is not supported.*Nx\.to_binary/, fn ->
        t = Nx.tensor([1, 2, 3])
        Mlx.Backend.to_pointer(t, [])
      end
    end
  end
end
