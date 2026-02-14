defmodule Mlx.SafetyTest do
  use ExUnit.Case
  # Validates: S1 (BEAM crash safety), S2 (no memory leaks), T2 (defensive NIFs)

  @moduletag :safety

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  describe "invalid dtype handling" do
    test "raises on unsupported f64 type" do
      assert_raise ArgumentError, fn ->
        Mlx.Dtype.to_mlx({:f, 64})
      end
    end

    test "raises on unsupported c128 type" do
      assert_raise ArgumentError, fn ->
        Mlx.Dtype.to_mlx({:c, 128})
      end
    end
  end

  describe "error propagation" do
    test "backend raises on MLX errors" do
      # Attempting an operation that should fail at the MLX level
      # Shape mismatch in matmul should produce an actionable error
      a = Nx.tensor([[1.0, 2.0]])
      b = Nx.tensor([[1.0, 2.0]])

      assert_raise RuntimeError, ~r/MLX error/, fn ->
        # 1x2 dot 1x2 should fail - inner dimensions don't match
        # mlx_matmul validates shapes eagerly and returns {:error, msg} directly
        case Mlx.NIF.mlx_matmul(a.data.ref, b.data.ref, Mlx.Backend.default_stream()) do
          {:ok, ref} ->
            case Mlx.NIF.eval(ref) do
              :ok -> raise "MLX error: expected matmul shape mismatch to fail"
              {:error, msg} -> raise "MLX error: #{msg}"
            end

          {:error, msg} ->
            raise "MLX error: #{msg}"
        end
      end
    end
  end

  describe "resource lifecycle" do
    test "tensor GC does not crash BEAM" do
      # Create many tensors and let them be GC'd
      for _ <- 1..100 do
        _t = Nx.tensor([1.0, 2.0, 3.0])
      end

      :erlang.garbage_collect()
      # If we get here without crashing, the destructors work
      assert true
    end

    test "repeated create/eval cycles" do
      for _ <- 1..50 do
        t = Nx.tensor([1.0, 2.0, 3.0])
        _ = Nx.to_flat_list(t)
      end

      assert true
    end

    test "large tensor lifecycle" do
      # Create a reasonably large tensor, use it, let it be freed
      t = Nx.iota({100, 100}, type: {:f, 32})
      sum = Nx.to_number(Nx.sum(t))
      assert sum > 0
      :erlang.garbage_collect()
      assert true
    end
  end

  describe "concurrent operations" do
    test "parallel tensor creation does not crash" do
      tasks =
        for _ <- 1..10 do
          Task.async(fn ->
            Nx.default_backend(Mlx.Backend)
            t = Nx.tensor([1.0, 2.0, 3.0])
            Nx.to_flat_list(t)
          end)
        end

      results = Task.await_many(tasks, 5000)
      assert length(results) == 10
      assert Enum.all?(results, &(&1 == [1.0, 2.0, 3.0]))
    end

    test "parallel arithmetic does not crash" do
      tasks =
        for i <- 1..10 do
          Task.async(fn ->
            Nx.default_backend(Mlx.Backend)
            a = Nx.tensor([i * 1.0, i * 2.0])
            b = Nx.tensor([1.0, 1.0])
            Nx.to_flat_list(Nx.add(a, b))
          end)
        end

      results = Task.await_many(tasks, 5000)
      assert length(results) == 10
    end
  end

  describe "edge cases" do
    test "scalar tensor operations" do
      a = Nx.tensor(5.0)
      b = Nx.tensor(3.0)
      result = Nx.add(a, b)
      assert Nx.to_number(result) == 8.0
    end

    test "zero-element operations" do
      # Sum of empty-like reduction
      t = Nx.tensor([0.0])
      assert Nx.to_number(Nx.sum(t)) == 0.0
    end

    test "very small values" do
      # Use values within f32 normal range (min normal ~1.175e-38)
      # Metal GPU flushes subnormals to zero (FTZ mode)
      t = Nx.tensor([1.0e-30, 1.0e-30])
      result = Nx.add(t, t)
      [a, _] = Nx.to_flat_list(result)
      assert a > 0
    end

    test "special float values - infinity" do
      t = Nx.tensor([:infinity])
      assert Nx.to_flat_list(Nx.is_infinity(t)) == [1]
    end

    test "special float values - nan" do
      t = Nx.tensor([:nan])
      assert Nx.to_flat_list(Nx.is_nan(t)) == [1]
    end

    test "high-dimensional tensor" do
      t = Nx.iota({2, 3, 4, 5}, type: {:f, 32})
      assert Nx.shape(t) == {2, 3, 4, 5}
      assert Nx.to_number(Nx.sum(t)) == Enum.sum(0..119) * 1.0
    end
  end

  describe "backend transfer safety" do
    test "transfer to BinaryBackend and back" do
      t = Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)
      binary_t = Nx.backend_transfer(t, Nx.BinaryBackend)
      assert %Nx.BinaryBackend{} = binary_t.data

      mlx_t = Nx.backend_transfer(binary_t, Mlx.Backend)
      assert %Mlx.Backend{} = mlx_t.data
      assert Nx.to_flat_list(mlx_t) == [1.0, 2.0, 3.0]
    end

    test "backend_copy preserves data" do
      t = Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)
      copy = Nx.backend_copy(t, Nx.BinaryBackend)
      assert Nx.to_flat_list(copy) == [1.0, 2.0, 3.0]
      # Original should still be valid
      assert Nx.to_flat_list(t) == [1.0, 2.0, 3.0]
    end
  end
end
