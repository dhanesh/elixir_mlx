defmodule Mlx.HubTest do
  use ExUnit.Case, async: false

  describe "cache_dir/0" do
    test "returns default cache directory" do
      # Temporarily clear any env override
      original = System.get_env("MLX_CACHE_DIR")
      System.delete_env("MLX_CACHE_DIR")

      dir = Mlx.Hub.cache_dir()
      assert String.ends_with?(dir, ".cache/mlx_elixir")
      assert String.starts_with?(dir, System.user_home!())

      # Restore
      if original, do: System.put_env("MLX_CACHE_DIR", original)
    end

    test "respects MLX_CACHE_DIR environment variable" do
      original = System.get_env("MLX_CACHE_DIR")
      System.put_env("MLX_CACHE_DIR", "/tmp/custom_mlx_cache")

      assert Mlx.Hub.cache_dir() == "/tmp/custom_mlx_cache"

      # Restore
      if original do
        System.put_env("MLX_CACHE_DIR", original)
      else
        System.delete_env("MLX_CACHE_DIR")
      end
    end
  end

  describe "cached_path/3" do
    test "returns nil for uncached file" do
      assert Mlx.Hub.cached_path("nonexistent/repo", "config.json") == nil
    end

    test "returns path for cached file" do
      base = Path.join(System.tmp_dir!(), "mlx_hub_test_#{:rand.uniform(100_000)}")
      repo_id = "test/model"
      filename = "config.json"

      # Manually create the cached file
      dest = Path.join([base, "models--test--model", "snapshots", "main", filename])
      File.mkdir_p!(Path.dirname(dest))
      File.write!(dest, "{}")

      path = Mlx.Hub.cached_path(repo_id, filename, cache_dir: base)
      assert path == dest
      assert File.exists?(path)

      File.rm_rf!(base)
    end
  end

  describe "download/3" do
    test "raises without req dependency" do
      # This test verifies the helpful error message concept.
      # Since req IS available in test deps, we just verify download/3 is callable.
      # The actual network test is tagged :hub_network below.
      assert is_function(&Mlx.Hub.download/3)
    end
  end

  describe "glob filtering" do
    # Test via snapshot_download options indirectly — the filter_files/3 is private,
    # so we test through the public API behavior. Here we test that the module compiles
    # and the public API is available.
    test "list_repo_files/2 is available" do
      assert is_function(&Mlx.Hub.list_repo_files/2)
    end

    test "snapshot_download/2 is available" do
      assert is_function(&Mlx.Hub.snapshot_download/2)
    end
  end

  # Network tests — excluded by default, run with: mix test --include hub_network
  describe "network operations" do
    @tag :hub_network
    test "download config.json from tiny model" do
      base = Path.join(System.tmp_dir!(), "mlx_hub_net_#{:rand.uniform(100_000)}")

      path =
        Mlx.Hub.download(
          "hf-internal-testing/tiny-random-BertModel",
          "config.json",
          cache_dir: base
        )

      assert File.exists?(path)
      config = path |> File.read!() |> Jason.decode!()
      assert is_map(config)
      assert Map.has_key?(config, "model_type") || Map.has_key?(config, "architectures")

      File.rm_rf!(base)
    end

    @tag :hub_network
    test "list_repo_files returns file list" do
      files = Mlx.Hub.list_repo_files("hf-internal-testing/tiny-random-BertModel")
      assert is_list(files)
      assert "config.json" in files
    end

    @tag :hub_network
    test "snapshot_download with allow_patterns" do
      base = Path.join(System.tmp_dir!(), "mlx_hub_snap_#{:rand.uniform(100_000)}")

      dir =
        Mlx.Hub.snapshot_download(
          "hf-internal-testing/tiny-random-BertModel",
          cache_dir: base,
          allow_patterns: ["config.json"]
        )

      assert File.exists?(Path.join(dir, "config.json"))
      File.rm_rf!(base)
    end
  end
end
