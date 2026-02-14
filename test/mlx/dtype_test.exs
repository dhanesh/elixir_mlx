defmodule Mlx.DtypeTest do
  use ExUnit.Case, async: true
  # Validates: RT-7 (dtype mapping)

  describe "to_mlx/1" do
    test "maps standard Nx types" do
      assert Mlx.Dtype.to_mlx({:f, 32}) == :f32
      assert Mlx.Dtype.to_mlx({:f, 16}) == :f16
      assert Mlx.Dtype.to_mlx({:bf, 16}) == :bf16
      assert Mlx.Dtype.to_mlx({:s, 32}) == :s32
      assert Mlx.Dtype.to_mlx({:s, 64}) == :s64
      assert Mlx.Dtype.to_mlx({:u, 8}) == :u8
      assert Mlx.Dtype.to_mlx({:u, 32}) == :u32
      assert Mlx.Dtype.to_mlx({:c, 64}) == :c64
    end

    test "maps pred type to bool" do
      assert Mlx.Dtype.to_mlx({:pred, 8}) == :bool
    end
  end

  describe "to_nx/1" do
    test "maps MLX types back to Nx" do
      assert Mlx.Dtype.to_nx(:f32) == {:f, 32}
      assert Mlx.Dtype.to_nx(:f16) == {:f, 16}
      assert Mlx.Dtype.to_nx(:bf16) == {:bf, 16}
      assert Mlx.Dtype.to_nx(:s32) == {:s, 32}
      assert Mlx.Dtype.to_nx(:u8) == {:u, 8}
      assert Mlx.Dtype.to_nx(:c64) == {:c, 64}
    end

    test "maps bool to u8" do
      assert Mlx.Dtype.to_nx(:bool) == {:u, 8}
    end
  end

  describe "roundtrip" do
    test "to_mlx -> to_nx preserves type" do
      for type <- [{:f, 32}, {:f, 16}, {:bf, 16}, {:s, 32}, {:s, 64}, {:u, 8}, {:u, 32}] do
        assert type == type |> Mlx.Dtype.to_mlx() |> Mlx.Dtype.to_nx()
      end
    end
  end

  describe "supported?/1" do
    test "f64 is not supported (Metal limitation)" do
      refute Mlx.Dtype.supported?({:f, 64})
    end

    test "c128 is not supported" do
      refute Mlx.Dtype.supported?({:c, 128})
    end

    test "f32 is supported" do
      assert Mlx.Dtype.supported?({:f, 32})
    end
  end

  describe "byte_size/1" do
    test "returns correct sizes" do
      assert Mlx.Dtype.byte_size(:f32) == 4
      assert Mlx.Dtype.byte_size(:f16) == 2
      assert Mlx.Dtype.byte_size(:s64) == 8
      assert Mlx.Dtype.byte_size(:u8) == 1
      assert Mlx.Dtype.byte_size(:bool) == 1
      assert Mlx.Dtype.byte_size(:c64) == 8
    end
  end
end
