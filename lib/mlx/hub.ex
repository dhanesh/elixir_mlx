defmodule Mlx.Hub do
  @moduledoc """
  HuggingFace Hub integration for downloading and caching model files.

  Provides local caching of downloaded models to avoid repeated downloads.
  Requires the optional `req` dependency for HTTP operations.

  ## Cache Structure

  Files are cached under `~/.cache/mlx_elixir/` (configurable via `MLX_CACHE_DIR`):

      ~/.cache/mlx_elixir/models--{owner}--{repo}/snapshots/{revision}/{filename}

  ## Authentication

  For private/gated models, set the `HF_TOKEN` environment variable or pass
  `:token` in options.

  ## Example

      # Download a single file
      path = Mlx.Hub.download("bert-base-uncased", "config.json")

      # Download all safetensors files
      dir = Mlx.Hub.snapshot_download("bert-base-uncased",
        allow_patterns: ["*.safetensors", "*.json"]
      )
  """

  @hf_api_url "https://huggingface.co/api/models"
  @hf_resolve_url "https://huggingface.co"

  @doc """
  Returns the cache directory for MLX models.

  Uses `MLX_CACHE_DIR` environment variable if set, otherwise defaults
  to `~/.cache/mlx_elixir`.
  """
  def cache_dir do
    System.get_env("MLX_CACHE_DIR") ||
      Path.join(System.user_home!(), ".cache/mlx_elixir")
  end

  @doc """
  Check if a file is already cached locally. Returns the path if cached, `nil` otherwise.

  ## Options
    * `:revision` - git revision (default: `"main"`)
    * `:cache_dir` - override cache directory
  """
  def cached_path(repo_id, filename, opts \\ []) do
    revision = Keyword.get(opts, :revision, "main")
    base = Keyword.get(opts, :cache_dir, cache_dir())
    path = snapshot_file_path(base, repo_id, revision, filename)

    if File.exists?(path), do: path, else: nil
  end

  @doc """
  Download a single file from a HuggingFace repository.

  Returns the local file path after downloading (or from cache if already present).

  ## Options
    * `:revision` - git revision (default: `"main"`)
    * `:token` - HuggingFace API token (default: `HF_TOKEN` env var)
    * `:cache_dir` - override cache directory
    * `:force_download` - re-download even if cached (default: `false`)
  """
  def download(repo_id, filename, opts \\ []) do
    ensure_req!()
    revision = Keyword.get(opts, :revision, "main")
    token = resolve_token(opts)
    base = Keyword.get(opts, :cache_dir, cache_dir())
    force = Keyword.get(opts, :force_download, false)

    dest = snapshot_file_path(base, repo_id, revision, filename)

    if !force && File.exists?(dest) do
      dest
    else
      url = build_url(repo_id, revision, filename)
      download_file!(url, dest, token)
      dest
    end
  end

  @doc """
  Download all (or matching) files from a HuggingFace repository.

  Returns the local snapshot directory path.

  ## Options
    * `:revision` - git revision (default: `"main"`)
    * `:token` - HuggingFace API token (default: `HF_TOKEN` env var)
    * `:cache_dir` - override cache directory
    * `:force_download` - re-download even if cached (default: `false`)
    * `:allow_patterns` - list of glob patterns to include (e.g. `["*.safetensors"]`)
    * `:ignore_patterns` - list of glob patterns to exclude
  """
  def snapshot_download(repo_id, opts \\ []) do
    ensure_req!()
    revision = Keyword.get(opts, :revision, "main")
    token = resolve_token(opts)
    base = Keyword.get(opts, :cache_dir, cache_dir())
    force = Keyword.get(opts, :force_download, false)
    allow = Keyword.get(opts, :allow_patterns, [])
    ignore = Keyword.get(opts, :ignore_patterns, [])

    files = list_repo_files(repo_id, token: token, revision: revision)
    files = filter_files(files, allow, ignore)

    snap_dir = snapshot_dir(base, repo_id, revision)

    for filename <- files do
      dest = Path.join(snap_dir, filename)

      if force || !File.exists?(dest) do
        url = build_url(repo_id, revision, filename)
        download_file!(url, dest, token)
      end
    end

    snap_dir
  end

  @doc """
  List files in a HuggingFace repository.

  ## Options
    * `:revision` - git revision (default: `"main"`)
    * `:token` - HuggingFace API token
  """
  def list_repo_files(repo_id, opts \\ []) do
    ensure_req!()
    token = resolve_token(opts)

    url = "#{@hf_api_url}/#{repo_id}"
    headers = auth_headers(token)

    response = Req.get!(url, headers: headers)

    case response.status do
      200 ->
        response.body
        |> Map.get("siblings", [])
        |> Enum.map(& &1["rfilename"])
        |> Enum.reject(&is_nil/1)

      status ->
        raise "HuggingFace API error (status #{status}) for #{repo_id}: #{inspect(response.body)}"
    end
  end

  # Private helpers

  defp ensure_req! do
    unless Code.ensure_loaded?(Req) do
      raise """
      Req is required for HuggingFace Hub downloads.
      Add {:req, "~> 0.5"} to your dependencies in mix.exs.
      """
    end
  end

  defp resolve_token(opts) do
    Keyword.get(opts, :token) || System.get_env("HF_TOKEN")
  end

  defp build_url(repo_id, revision, filename) do
    "#{@hf_resolve_url}/#{repo_id}/resolve/#{revision}/#{filename}"
  end

  defp repo_dir(base, repo_id) do
    safe_name = String.replace(repo_id, "/", "--")
    Path.join(base, "models--#{safe_name}")
  end

  defp snapshot_dir(base, repo_id, revision) do
    Path.join(repo_dir(base, repo_id), "snapshots/#{revision}")
  end

  defp snapshot_file_path(base, repo_id, revision, filename) do
    Path.join(snapshot_dir(base, repo_id, revision), filename)
  end

  defp auth_headers(nil), do: []
  defp auth_headers(token), do: [{"authorization", "Bearer #{token}"}]

  defp download_file!(url, dest, token) do
    File.mkdir_p!(Path.dirname(dest))
    headers = auth_headers(token)

    response = Req.get!(url, headers: headers, into: File.stream!(dest))

    case response.status do
      status when status in 200..299 ->
        :ok

      status ->
        File.rm(dest)
        raise "Failed to download #{url} (status #{status})"
    end
  end

  defp filter_files(files, [], []), do: files

  defp filter_files(files, allow, ignore) do
    files
    |> then(fn fs ->
      if allow == [] do
        fs
      else
        Enum.filter(fs, fn f -> Enum.any?(allow, &glob_match?(f, &1)) end)
      end
    end)
    |> then(fn fs ->
      if ignore == [] do
        fs
      else
        Enum.reject(fs, fn f -> Enum.any?(ignore, &glob_match?(f, &1)) end)
      end
    end)
  end

  defp glob_match?(filename, pattern) do
    # Convert glob pattern to regex
    regex_str =
      pattern
      |> String.replace(".", "\\.")
      |> String.replace("**", "<<GLOBSTAR>>")
      |> String.replace("*", "[^/]*")
      |> String.replace("<<GLOBSTAR>>", ".*")
      |> String.replace("?", "[^/]")

    Regex.match?(~r/^#{regex_str}$/, filename)
  end
end
