defmodule Mlx.Dtype do
  @moduledoc false
  # Satisfies: RT-7 (dtype mapping Nx <-> MLX)
  # Maps Nx type tuples like {:f, 32} to MLX atoms like :f32

  @nx_to_mlx %{
    {:u, 8} => :u8,
    {:u, 16} => :u16,
    {:u, 32} => :u32,
    {:u, 64} => :u64,
    {:s, 8} => :s8,
    {:s, 16} => :s16,
    {:s, 32} => :s32,
    {:s, 64} => :s64,
    {:f, 16} => :f16,
    {:f, 32} => :f32,
    {:bf, 16} => :bf16,
    {:c, 64} => :c64
  }

  @mlx_to_nx Map.new(@nx_to_mlx, fn {k, v} -> {v, k} end)

  @doc "Convert Nx type tuple to MLX dtype atom."
  def to_mlx({:pred, 8}), do: :bool

  def to_mlx(nx_type) do
    case Map.fetch(@nx_to_mlx, nx_type) do
      {:ok, mlx_type} ->
        mlx_type

      :error ->
        raise ArgumentError,
              "unsupported Nx type: #{inspect(nx_type)} (MLX/Metal does not support this type)"
    end
  end

  @doc "Convert MLX dtype atom to Nx type tuple."
  def to_nx(:bool), do: {:u, 8}

  def to_nx(mlx_dtype) do
    Map.fetch!(@mlx_to_nx, mlx_dtype)
  end

  @doc "Size in bytes for an MLX dtype."
  def byte_size(:bool), do: 1
  def byte_size(:u8), do: 1
  def byte_size(:u16), do: 2
  def byte_size(:u32), do: 4
  def byte_size(:u64), do: 8
  def byte_size(:s8), do: 1
  def byte_size(:s16), do: 2
  def byte_size(:s32), do: 4
  def byte_size(:s64), do: 8
  def byte_size(:f16), do: 2
  def byte_size(:f32), do: 4
  def byte_size(:bf16), do: 2
  def byte_size(:c64), do: 8

  @doc "Returns true if the Nx type is supported by MLX (Metal has no f64)."
  def supported?({:f, 64}), do: false
  def supported?({:c, 128}), do: false
  def supported?(_), do: true
end
