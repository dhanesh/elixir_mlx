defmodule ElixirMlx.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/elixir-mlx/elixir_mlx"
  @mlx_c_version "0.1.2"

  def project do
    [
      app: :elixir_mlx,
      version: @version,
      elixir: "~> 1.16",
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_targets: ["all"],
      make_clean: ["clean"],
      start_permanent: Mix.env() == :prod,
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
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
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
      files: ~w(lib c_src mix.exs Makefile README.md LICENSE)
    ]
  end

  defp docs do
    [
      main: "Mlx",
      extras: ["README.md"]
    ]
  end
end
