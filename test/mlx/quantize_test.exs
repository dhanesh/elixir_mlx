defmodule Mlx.QuantizeTest do
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 0.1)
    a_list = Nx.to_flat_list(a)
    b_list = Nx.to_flat_list(b)

    Enum.zip(a_list, b_list)
    |> Enum.each(fn {av, bv} ->
      assert_in_delta(av, bv, atol, "expected #{av} to be close to #{bv}")
    end)
  end

  defp make_matrix(rows, cols) do
    # Deterministic "random-looking" matrix using iota + math
    flat = for i <- 0..(rows * cols - 1), do: :math.sin(i / 1.0) * 0.5
    Nx.tensor(Enum.chunk_every(flat, cols))
  end

  describe "quantize/dequantize" do
    test "quantize returns 3 components" do
      w = make_matrix(4, 32)
      {quantized, scales, biases} = Mlx.Quantize.quantize(w, group_size: 32, bits: 4)
      assert is_struct(quantized, Nx.Tensor)
      assert is_struct(scales, Nx.Tensor)
      assert is_struct(biases, Nx.Tensor)
    end

    test "roundtrip approximately preserves values" do
      w = make_matrix(4, 32)
      {quantized, scales, biases} = Mlx.Quantize.quantize(w, group_size: 32, bits: 8)
      recovered = Mlx.Quantize.dequantize(quantized, scales, biases, group_size: 32, bits: 8)
      # 8-bit quantization should be fairly close
      assert_all_close(recovered, w, atol: 0.05)
    end
  end

  describe "quantized_matmul" do
    test "produces correct shape" do
      w = make_matrix(32, 32)
      {quantized, scales, biases} = Mlx.Quantize.quantize(w, group_size: 32, bits: 4)
      x = make_matrix(1, 32)

      result =
        Mlx.Quantize.quantized_matmul(x, quantized, scales, biases, group_size: 32, bits: 4)

      assert Nx.shape(result) == {1, 32}
    end
  end
end
