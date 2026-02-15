defmodule ElixirMlx.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/elixir-mlx/elixir_mlx"
  def project do
    [
      app: :elixir_mlx,
      version: @version,
      elixir: "~> 1.16",
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_targets: ["all"],
      make_clean: ["clean"],
      make_precompiler: {:nif, CCPrecompiler},
      make_precompiler_url:
        "https://github.com/elixir-mlx/elixir_mlx/releases/download/v#{@version}/@{artefact_filename}",
      make_precompiler_filename: "mlx_nif",
      make_precompiler_priv_paths: ["mlx_nif.*"],
      make_precompiler_nif_versions: [versions: ["2.16", "2.17"]],
      cc_precompiler: [
        only_listed_targets: true,
        compilers: %{
          {:unix, :darwin} => %{
            "aarch64-apple-darwin" => {"gcc", "g++"}
          }
        }
      ],
      start_permanent: Mix.env() == :prod,
      test_coverage: [
        ignore_modules: [Mlx.NIF]
      ],
      deps: deps(),
      name: "ElixirMlx",
      description: description(),
      package: package(),
      docs: docs(),
      source_url: @source_url
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.9"},
      {:elixir_make, "~> 0.8", runtime: false},
      {:cc_precompiler, "~> 0.1", runtime: false},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:axon, "~> 0.8", only: :test},
      {:polaris, "~> 0.1", only: :test},
      {:scholar, "~> 0.4", only: :test}
    ]
  end

  defp description do
    """
    MLX bindings for Elixir via mlx-c. An Nx backend and Nx.Defn compiler
    for Apple's MLX machine learning framework on Apple Silicon.
    """
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib c_src mix.exs Makefile README.md CHANGELOG.md LICENSE checksum-elixir_mlx.exs)
    ]
  end

  defp docs do
    [
      main: "Mlx",
      extras: ["README.md", "CHANGELOG.md"]
    ]
  end
end
