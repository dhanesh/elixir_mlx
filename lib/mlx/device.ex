defmodule Mlx.Device do
  @moduledoc """
  Device management for MLX.
  Satisfies: RT-2 (device management), T6 (unified memory)

  MLX uses Apple's unified memory architecture, so CPU and GPU
  share the same memory space. No data copies are needed when
  switching devices — only the execution target changes.
  """

  @doc "Returns the default device (typically GPU on Apple Silicon)."
  def default do
    case Mlx.NIF.default_device() do
      {:ok, ref} -> ref
      {:error, reason} -> raise "failed to get default device: #{reason}"
    end
  end

  @doc "Creates a device of the given type (:cpu or :gpu)."
  def new(type) when type in [:cpu, :gpu] do
    case Mlx.NIF.device_new(type) do
      {:ok, ref} -> ref
      {:error, reason} -> raise "failed to create #{type} device: #{reason}"
    end
  end

  @doc "Sets the default device."
  def set_default(type) when type in [:cpu, :gpu] do
    device = new(type)

    case Mlx.NIF.set_default_device(device) do
      :ok -> :ok
      {:error, reason} -> raise "failed to set default device: #{reason}"
    end
  end
end
