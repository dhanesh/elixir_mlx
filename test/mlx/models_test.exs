defmodule Mlx.ModelsTest do
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  defp assert_all_close(a, b, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-4)
    a_list = Nx.to_flat_list(a)
    b_list = Nx.to_flat_list(b)

    Enum.zip(a_list, b_list)
    |> Enum.each(fn {av, bv} ->
      assert_in_delta(av, bv, atol, "expected #{av} to be close to #{bv}")
    end)
  end

  describe "unflatten_params/1" do
    test "flat dot-separated keys become nested map" do
      flat = %{"a.b.c" => 1, "a.b.d" => 2, "x" => 3}
      nested = Mlx.Models.unflatten_params(flat)

      assert nested == %{"a" => %{"b" => %{"c" => 1, "d" => 2}}, "x" => 3}
    end

    test "single-level keys remain unchanged" do
      flat = %{"weight" => 1, "bias" => 2}
      assert Mlx.Models.unflatten_params(flat) == flat
    end

    test "deep nesting works" do
      flat = %{"a.b.c.d.e" => 42}
      nested = Mlx.Models.unflatten_params(flat)
      assert nested == %{"a" => %{"b" => %{"c" => %{"d" => %{"e" => 42}}}}}
    end

    test "empty map returns empty map" do
      assert Mlx.Models.unflatten_params(%{}) == %{}
    end
  end

  describe "load_config/1" do
    test "loads config from directory" do
      dir = Path.join(System.tmp_dir!(), "mlx_models_cfg_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      File.write!(
        Path.join(dir, "config.json"),
        Jason.encode!(%{"model_type" => "test", "hidden_size" => 128})
      )

      config = Mlx.Models.load_config(dir)
      assert config["model_type"] == "test"
      assert config["hidden_size"] == 128

      File.rm_rf!(dir)
    end

    test "loads config from direct path" do
      path = Path.join(System.tmp_dir!(), "mlx_models_cfg_direct_#{:rand.uniform(100_000)}.json")
      File.write!(path, Jason.encode!(%{"vocab_size" => 30522}))

      config = Mlx.Models.load_config(path)
      assert config["vocab_size"] == 30522

      File.rm(path)
    end

    test "raises on missing config" do
      assert_raise ArgumentError, ~r/config file not found/, fn ->
        Mlx.Models.load_config("/tmp/nonexistent_mlx_dir_#{:rand.uniform(100_000)}")
      end
    end
  end

  describe "to_model_state/1,2" do
    test "nested map creates ModelState" do
      weights = %{
        "dense" => %{
          "kernel" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
          "bias" => Nx.tensor([0.1, 0.2])
        }
      }

      state = Mlx.Models.to_model_state(weights, unflatten: false)
      assert %Axon.ModelState{} = state
      assert Map.has_key?(state.data, "dense")
      assert Map.has_key?(state.data["dense"], "kernel")
    end

    test "flat map with unflatten creates nested ModelState" do
      weights = %{
        "layer.weight" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "layer.bias" => Nx.tensor([0.5, 1.5])
      }

      state = Mlx.Models.to_model_state(weights)
      assert %Axon.ModelState{} = state
      assert Map.has_key?(state.data, "layer")
      assert Map.has_key?(state.data["layer"], "weight")
      assert Map.has_key?(state.data["layer"], "bias")
    end

    test "with remap_fn transforms keys" do
      weights = %{
        "old_name.weight" => Nx.tensor([1.0, 2.0])
      }

      remap = fn key -> String.replace(key, "old_name", "new_name") end
      state = Mlx.Models.to_model_state(weights, remap_fn: remap)

      assert %Axon.ModelState{} = state
      assert Map.has_key?(state.data, "new_name")
    end
  end

  describe "load_model_state/2" do
    test "loads from local directory with safetensors" do
      dir = Path.join(System.tmp_dir!(), "mlx_models_lms_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      tensors = %{
        "encoder.0.weight" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "encoder.0.bias" => Nx.tensor([0.1, 0.2])
      }

      Mlx.IO.save_safetensors(Path.join(dir, "model.safetensors"), tensors)
      File.write!(Path.join(dir, "config.json"), Jason.encode!(%{"model_type" => "test"}))

      state = Mlx.Models.load_model_state(dir)
      assert %Axon.ModelState{} = state

      # Verify the unflattened structure
      assert Map.has_key?(state.data, "encoder")
      assert Map.has_key?(state.data["encoder"], "0")
      assert Map.has_key?(state.data["encoder"]["0"], "weight")

      File.rm_rf!(dir)
    end

    test "loads with dtype casting" do
      dir = Path.join(System.tmp_dir!(), "mlx_models_dtype_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      tensors = %{"w" => Nx.tensor([1.0, 2.0, 3.0])}
      Mlx.IO.save_safetensors(Path.join(dir, "model.safetensors"), tensors)

      state = Mlx.Models.load_model_state(dir, dtype: {:f, 16})
      assert %Axon.ModelState{} = state
      assert Nx.type(state.data["w"]) == {:f, 16}

      File.rm_rf!(dir)
    end

    test "loads with unflatten disabled" do
      dir = Path.join(System.tmp_dir!(), "mlx_models_nouf_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      tensors = %{"layer.weight" => Nx.tensor([1.0, 2.0])}
      Mlx.IO.save_safetensors(Path.join(dir, "model.safetensors"), tensors)

      state = Mlx.Models.load_model_state(dir, unflatten: false)
      assert %Axon.ModelState{} = state
      # Keys should remain flat
      assert Map.has_key?(state.data, "layer.weight")

      File.rm_rf!(dir)
    end

    test "loads with remap_fn" do
      dir = Path.join(System.tmp_dir!(), "mlx_models_remap_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      tensors = %{"transformer.h.0.weight" => Nx.tensor([1.0, 2.0])}
      Mlx.IO.save_safetensors(Path.join(dir, "model.safetensors"), tensors)

      remap = fn key -> String.replace(key, "transformer.h", "blocks") end
      state = Mlx.Models.load_model_state(dir, remap_fn: remap)

      assert %Axon.ModelState{} = state
      assert Map.has_key?(state.data, "blocks")

      File.rm_rf!(dir)
    end
  end

  describe "dequantize integration" do
    test "dequantizes weight triplets and removes scales/biases" do
      dir = Path.join(System.tmp_dir!(), "mlx_models_deq_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      # MLX requires minimum group_size of 32 — use a {1, 32} tensor
      original = Nx.tensor([Enum.map(1..32, &(&1 / 1))])
      {quantized, scales, biases} = Mlx.Quantize.quantize(original, group_size: 32, bits: 4)

      tensors = %{
        "layer.weight" => quantized,
        "layer.scales" => scales,
        "layer.biases" => biases,
        "other.weight" => Nx.tensor([10.0, 20.0])
      }

      Mlx.IO.save_safetensors(Path.join(dir, "model.safetensors"), tensors)

      File.write!(
        Path.join(dir, "quantize_config.json"),
        Jason.encode!(%{"group_size" => 32, "bits" => 4})
      )

      state = Mlx.Models.load_model_state(dir)
      assert %Axon.ModelState{} = state

      # The dequantized weight should be close to the original
      deq_weight = state.data["layer"]["weight"]
      assert_all_close(deq_weight, original, atol: 1.5)

      # scales and biases should be removed from the state
      layer_keys = Map.keys(state.data["layer"])
      refute "scales" in layer_keys
      refute "biases" in layer_keys

      # other.weight should remain untouched
      assert Map.has_key?(state.data, "other")

      File.rm_rf!(dir)
    end

    test "skips dequantization when disabled" do
      dir = Path.join(System.tmp_dir!(), "mlx_models_nodeq_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      original = Nx.tensor([Enum.map(1..32, &(&1 / 1))])
      {quantized, scales, biases} = Mlx.Quantize.quantize(original, group_size: 32, bits: 4)

      tensors = %{
        "layer.weight" => quantized,
        "layer.scales" => scales,
        "layer.biases" => biases
      }

      Mlx.IO.save_safetensors(Path.join(dir, "model.safetensors"), tensors)

      File.write!(
        Path.join(dir, "quantize_config.json"),
        Jason.encode!(%{"group_size" => 32, "bits" => 4})
      )

      state = Mlx.Models.load_model_state(dir, dequantize: false)
      assert %Axon.ModelState{} = state

      # scales and biases should still be present
      assert Map.has_key?(state.data["layer"], "scales")
      assert Map.has_key?(state.data["layer"], "biases")

      File.rm_rf!(dir)
    end
  end

  describe "end-to-end smoke test" do
    test "save and load via full pipeline produces ModelState" do
      dir = Path.join(System.tmp_dir!(), "mlx_e2e_#{:rand.uniform(100_000)}")
      File.mkdir_p!(dir)

      Mlx.IO.save_safetensors(Path.join(dir, "model.safetensors"), %{
        "encoder.0.weight" => Nx.iota({4, 4}, type: {:f, 32}),
        "encoder.0.bias" => Nx.tensor([0.1, 0.2, 0.3, 0.4])
      })

      File.write!(Path.join(dir, "config.json"), Jason.encode!(%{"model_type" => "test"}))

      state = Mlx.Models.load_model_state(dir)
      assert %Axon.ModelState{} = state

      # Verify nested structure
      assert Map.has_key?(state.data, "encoder")
      assert Map.has_key?(state.data["encoder"]["0"], "weight")
      assert Nx.shape(state.data["encoder"]["0"]["weight"]) == {4, 4}

      File.rm_rf!(dir)
    end
  end
end
