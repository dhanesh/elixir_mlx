defmodule Mlx.IOTest do
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    a_list = Nx.to_flat_list(a)
    b_list = Nx.to_flat_list(b)

    Enum.zip(a_list, b_list)
    |> Enum.each(fn {av, bv} ->
      assert_in_delta(av, bv, atol, "expected #{av} to be close to #{bv}")
    end)
  end

  describe "save/load (npy)" do
    test "roundtrip 1D tensor" do
      path = Path.join(System.tmp_dir!(), "mlx_test_1d.npy")
      a = Nx.tensor([1.0, 2.0, 3.0])
      Mlx.IO.save(path, a)
      loaded = Mlx.IO.load(path)
      assert_all_close(loaded, a)
      File.rm(path)
    end

    test "roundtrip 2D tensor" do
      path = Path.join(System.tmp_dir!(), "mlx_test_2d.npy")
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      Mlx.IO.save(path, a)
      loaded = Mlx.IO.load(path)
      assert Nx.shape(loaded) == {2, 2}
      assert_all_close(loaded, a)
      File.rm(path)
    end
  end

  describe "save_safetensors/load_safetensors" do
    test "roundtrip named tensors" do
      path = Path.join(System.tmp_dir!(), "mlx_test.safetensors")

      tensors = %{
        "weight" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "bias" => Nx.tensor([0.5, 1.5])
      }

      Mlx.IO.save_safetensors(path, tensors)
      {loaded, _metadata} = Mlx.IO.load_safetensors(path)

      assert Map.has_key?(loaded, "weight")
      assert Map.has_key?(loaded, "bias")
      assert_all_close(loaded["weight"], tensors["weight"])
      assert_all_close(loaded["bias"], tensors["bias"])
      File.rm(path)
    end

    test "roundtrip with metadata" do
      path = Path.join(System.tmp_dir!(), "mlx_test_meta.safetensors")

      tensors = %{"data" => Nx.tensor([1.0, 2.0, 3.0])}
      metadata = %{"format" => "mlx", "version" => "0.1"}

      Mlx.IO.save_safetensors(path, tensors, metadata: metadata)
      {loaded, loaded_meta} = Mlx.IO.load_safetensors(path)

      assert_all_close(loaded["data"], tensors["data"])
      assert loaded_meta["format"] == "mlx"
      assert loaded_meta["version"] == "0.1"
      File.rm(path)
    end
  end

  describe "load_weights/1,2" do
    test "load_weights with .npy file" do
      path = Path.join(System.tmp_dir!(), "mlx_lw_test.npy")
      a = Nx.tensor([1.0, 2.0, 3.0])
      Mlx.IO.save(path, a)

      weights = Mlx.IO.load_weights(path)
      assert is_map(weights)
      assert Map.has_key?(weights, "weights")
      assert_all_close(weights["weights"], a)
      File.rm(path)
    end

    test "load_weights with .safetensors file returns flat map" do
      path = Path.join(System.tmp_dir!(), "mlx_lw_test.safetensors")

      tensors = %{
        "layer.weight" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "layer.bias" => Nx.tensor([0.5, 1.5])
      }

      Mlx.IO.save_safetensors(path, tensors)
      weights = Mlx.IO.load_weights(path)

      assert is_map(weights)
      assert Map.has_key?(weights, "layer.weight")
      assert Map.has_key?(weights, "layer.bias")
      assert_all_close(weights["layer.weight"], tensors["layer.weight"])
      File.rm(path)
    end

    test "load_weights from directory with single model.safetensors" do
      dir = Path.join(System.tmp_dir!(), "mlx_lw_dir_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      tensors = %{
        "encoder.weight" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "encoder.bias" => Nx.tensor([0.1, 0.2])
      }

      Mlx.IO.save_safetensors(Path.join(dir, "model.safetensors"), tensors)
      weights = Mlx.IO.load_weights(dir)

      assert Map.has_key?(weights, "encoder.weight")
      assert Map.has_key?(weights, "encoder.bias")
      assert_all_close(weights["encoder.weight"], tensors["encoder.weight"])
      File.rm_rf!(dir)
    end

    test "load_weights from directory with sharded safetensors" do
      dir = Path.join(System.tmp_dir!(), "mlx_lw_sharded_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      # Create two shards
      shard1 = %{"layer0.weight" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]])}
      shard2 = %{"layer1.weight" => Nx.tensor([[5.0, 6.0], [7.0, 8.0]])}

      Mlx.IO.save_safetensors(Path.join(dir, "model-00001-of-00002.safetensors"), shard1)
      Mlx.IO.save_safetensors(Path.join(dir, "model-00002-of-00002.safetensors"), shard2)

      index = %{
        "metadata" => %{"total_size" => 128},
        "weight_map" => %{
          "layer0.weight" => "model-00001-of-00002.safetensors",
          "layer1.weight" => "model-00002-of-00002.safetensors"
        }
      }

      File.write!(Path.join(dir, "model.safetensors.index.json"), Jason.encode!(index))

      weights = Mlx.IO.load_weights(dir)
      assert map_size(weights) == 2
      assert Map.has_key?(weights, "layer0.weight")
      assert Map.has_key?(weights, "layer1.weight")
      assert_all_close(weights["layer0.weight"], shard1["layer0.weight"])
      assert_all_close(weights["layer1.weight"], shard2["layer1.weight"])
      File.rm_rf!(dir)
    end

    test "load_weights with :dtype option casts all tensors" do
      path = Path.join(System.tmp_dir!(), "mlx_lw_dtype.safetensors")
      tensors = %{"w" => Nx.tensor([1.0, 2.0, 3.0])}
      Mlx.IO.save_safetensors(path, tensors)

      weights = Mlx.IO.load_weights(path, dtype: {:f, 16})
      assert Nx.type(weights["w"]) == {:f, 16}
      File.rm(path)
    end

    test "load_weights raises on missing directory contents" do
      dir = Path.join(System.tmp_dir!(), "mlx_lw_empty_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      assert_raise ArgumentError, ~r/does not contain/, fn ->
        Mlx.IO.load_weights(dir)
      end

      File.rm_rf!(dir)
    end

    test "load_weights raises on unrecognized format" do
      path = Path.join(System.tmp_dir!(), "mlx_lw_unknown.bin")
      File.write!(path, "fake")

      assert_raise ArgumentError, ~r/cannot detect format/, fn ->
        Mlx.IO.load_weights(path)
      end

      File.rm(path)
    end
  end
end
