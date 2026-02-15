defmodule Mlx.IOTest do
  use ExUnit.Case, async: true

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
end
