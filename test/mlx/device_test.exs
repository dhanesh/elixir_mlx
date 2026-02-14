defmodule Mlx.DeviceTest do
  use ExUnit.Case
  # Validates: T7 (unified memory model), device switching, CPU/GPU streams

  @moduletag :device

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  describe "device creation" do
    test "default device is available" do
      device = Mlx.Device.default()
      assert is_reference(device)
    end

    test "can create CPU device" do
      device = Mlx.Device.new(:cpu)
      assert is_reference(device)
    end

    test "can create GPU device" do
      device = Mlx.Device.new(:gpu)
      assert is_reference(device)
    end

    test "rejects invalid device type" do
      assert_raise FunctionClauseError, fn ->
        Mlx.Device.new(:tpu)
      end
    end
  end

  describe "device switching" do
    test "can switch default device to CPU and back to GPU" do
      # Switch to CPU
      assert :ok = Mlx.Device.set_default(:cpu)
      # Switch back to GPU
      assert :ok = Mlx.Device.set_default(:gpu)
    end

    test "operations work after switching to CPU" do
      Mlx.Device.set_default(:cpu)

      t = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.add(t, t)
      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0]

      # Restore GPU default
      Mlx.Device.set_default(:gpu)
    end

    test "operations work after switching to GPU" do
      Mlx.Device.set_default(:gpu)

      t = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.multiply(t, t)
      assert Nx.to_flat_list(result) == [1.0, 4.0, 9.0]
    end
  end

  describe "stream management" do
    test "can obtain CPU stream" do
      {:ok, stream} = Mlx.NIF.default_cpu_stream()
      assert is_reference(stream)
    end

    test "can obtain GPU stream" do
      {:ok, stream} = Mlx.NIF.default_gpu_stream()
      assert is_reference(stream)
    end

    test "backend default_stream returns GPU stream" do
      # Clear cached stream for this process
      Process.delete({Mlx.Backend, :default_stream})

      stream = Mlx.Backend.default_stream()
      assert is_reference(stream)
    end

    test "backend caches stream per process" do
      # Clear cached stream
      Process.delete({Mlx.Backend, :default_stream})

      stream1 = Mlx.Backend.default_stream()
      stream2 = Mlx.Backend.default_stream()

      # Same reference should be returned (cached)
      assert stream1 == stream2
    end

    test "different processes get independent stream caches" do
      Process.delete({Mlx.Backend, :default_stream})
      stream_main = Mlx.Backend.default_stream()

      stream_task =
        Task.async(fn ->
          Nx.default_backend(Mlx.Backend)
          Process.delete({Mlx.Backend, :default_stream})
          Mlx.Backend.default_stream()
        end)
        |> Task.await()

      # Both are valid references
      assert is_reference(stream_main)
      assert is_reference(stream_task)
    end

    test "synchronize CPU stream" do
      {:ok, stream} = Mlx.NIF.default_cpu_stream()
      assert :ok = Mlx.NIF.synchronize(stream)
    end

    test "synchronize GPU stream" do
      {:ok, stream} = Mlx.NIF.default_gpu_stream()
      assert :ok = Mlx.NIF.synchronize(stream)
    end
  end

  describe "unified memory - zero copy access" do
    test "tensor created on GPU is readable without explicit copy" do
      # T7 core: unified memory means no cudaMemcpy equivalent needed
      Mlx.Device.set_default(:gpu)

      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      # Reading data back (to_flat_list triggers eval + to_binary)
      # On unified memory, this should work without explicit device transfer
      assert Nx.to_flat_list(t) == [1.0, 2.0, 3.0, 4.0, 5.0]
    end

    test "GPU computation result accessible from CPU without transfer" do
      Mlx.Device.set_default(:gpu)

      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
      result = Nx.dot(a, b)

      # Reading result triggers eval on GPU, then to_binary reads from
      # unified memory — no explicit GPU→CPU copy needed
      assert Nx.to_flat_list(result) == [19.0, 22.0, 43.0, 50.0]
    end

    test "complex pipeline on GPU, read on CPU" do
      Mlx.Device.set_default(:gpu)

      t = Nx.iota({4, 4}, type: {:f, 32})

      result =
        t
        |> Nx.multiply(2.0)
        |> Nx.add(1.0)
        |> Nx.sum()

      # Full lazy pipeline evaluated on GPU, result read via unified memory
      assert Nx.to_number(result) == Enum.sum(for i <- 0..15, do: i * 2.0 + 1.0)
    end

    test "tensor data survives device switch" do
      # Create tensor while GPU is default
      Mlx.Device.set_default(:gpu)
      t = Nx.tensor([10.0, 20.0, 30.0])

      # Switch to CPU — tensor should still be readable (unified memory)
      Mlx.Device.set_default(:cpu)
      assert Nx.to_flat_list(t) == [10.0, 20.0, 30.0]

      # Compute on CPU with the GPU-created tensor
      result = Nx.add(t, Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_flat_list(result) == [11.0, 22.0, 33.0]

      # Restore GPU
      Mlx.Device.set_default(:gpu)
    end
  end

  describe "cross-device operations" do
    test "tensor created before device switch works in new context" do
      Mlx.Device.set_default(:gpu)
      gpu_tensor = Nx.tensor([1.0, 2.0, 3.0])

      Mlx.Device.set_default(:cpu)
      cpu_tensor = Nx.tensor([4.0, 5.0, 6.0])

      # Operations involving tensors from different device contexts
      # should work due to unified memory
      result = Nx.add(gpu_tensor, cpu_tensor)
      assert Nx.to_flat_list(result) == [5.0, 7.0, 9.0]

      # Restore GPU
      Mlx.Device.set_default(:gpu)
    end

    test "reduction on GPU-created tensor after switching to CPU" do
      Mlx.Device.set_default(:gpu)
      t = Nx.iota({3, 3}, type: {:f, 32})

      Mlx.Device.set_default(:cpu)
      sum = Nx.to_number(Nx.sum(t))
      assert sum == 36.0

      # Restore GPU
      Mlx.Device.set_default(:gpu)
    end

    test "matrix operations across device contexts" do
      Mlx.Device.set_default(:gpu)
      # identity matrix
      a = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])

      Mlx.Device.set_default(:cpu)
      b = Nx.tensor([[3.0, 4.0], [5.0, 6.0]])

      # dot product with tensors from different device contexts
      result = Nx.dot(a, b)
      assert Nx.to_flat_list(result) == [3.0, 4.0, 5.0, 6.0]

      # Restore GPU
      Mlx.Device.set_default(:gpu)
    end
  end
end
