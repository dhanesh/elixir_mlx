defmodule Mlx.WindowTest do
  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  describe "window_sum" do
    test "1D basic" do
      t = Nx.tensor([1, 2, 3, 4, 5])
      result = Nx.window_sum(t, {3})
      assert Nx.to_flat_list(result) == [6, 9, 12]
    end

    test "2D basic" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      result = Nx.window_sum(t, {2, 2})
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [12, 16, 24, 28]
    end

    test "with strides" do
      t = Nx.tensor([1, 2, 3, 4, 5, 6])
      result = Nx.window_sum(t, {2}, strides: [2])
      assert Nx.to_flat_list(result) == [3, 7, 11]
    end

    test "with padding" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.window_sum(t, {3}, padding: [{1, 1}])
      assert Nx.shape(result) == {3}
      expected = [3.0, 6.0, 5.0]
      result_list = Nx.to_flat_list(result)
      Enum.zip(expected, result_list)
      |> Enum.each(fn {e, r} -> assert_in_delta(e, r, 1.0e-5) end)
    end
  end

  describe "window_max" do
    test "1D basic" do
      t = Nx.tensor([1, 3, 2, 5, 4])
      result = Nx.window_max(t, {3})
      assert Nx.to_flat_list(result) == [3, 5, 5]
    end

    test "2D basic" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      result = Nx.window_max(t, {2, 2})
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [5, 6, 8, 9]
    end
  end

  describe "window_min" do
    test "1D basic" do
      t = Nx.tensor([5, 3, 4, 1, 2])
      result = Nx.window_min(t, {3})
      assert Nx.to_flat_list(result) == [3, 1, 1]
    end

    test "2D basic" do
      t = Nx.tensor([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
      result = Nx.window_min(t, {2, 2})
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [5, 4, 2, 1]
    end
  end

  describe "window_product" do
    test "1D basic" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = Nx.window_product(t, {2})
      result_list = Nx.to_flat_list(result)
      expected = [2.0, 6.0, 12.0]
      Enum.zip(expected, result_list)
      |> Enum.each(fn {e, r} -> assert_in_delta(e, r, 1.0e-5) end)
    end
  end

  describe "window with dilations" do
    test "window_sum with dilation" do
      t = Nx.tensor([1, 2, 3, 4, 5])
      # Window {2} with dilation [2] means elements at positions i and i+2
      result = Nx.window_sum(t, {2}, window_dilations: [2])
      # Effective window size: (2-1)*2 + 1 = 3
      # Output size: (5 - 3) / 1 + 1 = 3
      # Windows: [1,3], [2,4], [3,5]
      assert Nx.to_flat_list(result) == [4, 6, 8]
    end
  end
end
