defmodule Mlx.CoverageTest do
  @moduledoc """
  Targeted tests to improve coverage on modules with gaps:
  - Mlx (top-level API: eval, synchronize)
  - Mlx.Compiler (container traversal: structs, maps, lists)
  - Mlx.Dtype (all byte_size clauses, all mappings, error paths)
  - Mlx.Device (error path for set_default)
  """
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  # ── Mlx top-level API ──

  describe "Mlx.eval/1" do
    test "evaluates a lazy tensor" do
      t = Nx.add(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0]))
      result = Mlx.eval(t)
      assert %Nx.Tensor{} = result
      assert Nx.to_flat_list(result) == [4.0, 6.0]
    end

    test "evaluating an already-evaluated tensor is idempotent" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      _ = Nx.to_flat_list(t)
      result = Mlx.eval(t)
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end
  end

  describe "Mlx.synchronize/0" do
    test "synchronize with default stream" do
      _ = Nx.tensor([1.0, 2.0])
      assert :ok = Mlx.synchronize()
    end

    test "synchronize with explicit stream" do
      {:ok, stream} = Mlx.NIF.default_gpu_stream()
      assert :ok = Mlx.synchronize(stream)
    end
  end

  # ── Mlx.Dtype additional coverage ──

  describe "Mlx.Dtype.to_mlx/1 full coverage" do
    test "all integer types" do
      assert Mlx.Dtype.to_mlx({:u, 8}) == :u8
      assert Mlx.Dtype.to_mlx({:u, 16}) == :u16
      assert Mlx.Dtype.to_mlx({:u, 32}) == :u32
      assert Mlx.Dtype.to_mlx({:u, 64}) == :u64
      assert Mlx.Dtype.to_mlx({:s, 8}) == :s8
      assert Mlx.Dtype.to_mlx({:s, 16}) == :s16
      assert Mlx.Dtype.to_mlx({:s, 32}) == :s32
      assert Mlx.Dtype.to_mlx({:s, 64}) == :s64
    end

    test "all float types" do
      assert Mlx.Dtype.to_mlx({:f, 16}) == :f16
      assert Mlx.Dtype.to_mlx({:f, 32}) == :f32
      assert Mlx.Dtype.to_mlx({:bf, 16}) == :bf16
    end

    test "complex type" do
      assert Mlx.Dtype.to_mlx({:c, 64}) == :c64
    end

    test "pred type" do
      assert Mlx.Dtype.to_mlx({:pred, 8}) == :bool
    end

    test "unsupported type raises" do
      assert_raise ArgumentError, ~r/unsupported Nx type/, fn ->
        Mlx.Dtype.to_mlx({:f, 64})
      end
    end
  end

  describe "Mlx.Dtype.to_nx/1 full coverage" do
    test "all integer types" do
      assert Mlx.Dtype.to_nx(:u8) == {:u, 8}
      assert Mlx.Dtype.to_nx(:u16) == {:u, 16}
      assert Mlx.Dtype.to_nx(:u32) == {:u, 32}
      assert Mlx.Dtype.to_nx(:u64) == {:u, 64}
      assert Mlx.Dtype.to_nx(:s8) == {:s, 8}
      assert Mlx.Dtype.to_nx(:s16) == {:s, 16}
      assert Mlx.Dtype.to_nx(:s32) == {:s, 32}
      assert Mlx.Dtype.to_nx(:s64) == {:s, 64}
    end

    test "all float types" do
      assert Mlx.Dtype.to_nx(:f16) == {:f, 16}
      assert Mlx.Dtype.to_nx(:f32) == {:f, 32}
      assert Mlx.Dtype.to_nx(:bf16) == {:bf, 16}
    end

    test "bool type" do
      assert Mlx.Dtype.to_nx(:bool) == {:u, 8}
    end

    test "complex type" do
      assert Mlx.Dtype.to_nx(:c64) == {:c, 64}
    end
  end

  describe "Mlx.Dtype.byte_size/1 full coverage" do
    test "all byte sizes" do
      assert Mlx.Dtype.byte_size(:bool) == 1
      assert Mlx.Dtype.byte_size(:u8) == 1
      assert Mlx.Dtype.byte_size(:u16) == 2
      assert Mlx.Dtype.byte_size(:u32) == 4
      assert Mlx.Dtype.byte_size(:u64) == 8
      assert Mlx.Dtype.byte_size(:s8) == 1
      assert Mlx.Dtype.byte_size(:s16) == 2
      assert Mlx.Dtype.byte_size(:s32) == 4
      assert Mlx.Dtype.byte_size(:s64) == 8
      assert Mlx.Dtype.byte_size(:f16) == 2
      assert Mlx.Dtype.byte_size(:f32) == 4
      assert Mlx.Dtype.byte_size(:bf16) == 2
      assert Mlx.Dtype.byte_size(:c64) == 8
    end
  end

  describe "Mlx.Dtype.supported?/1" do
    test "all supported types" do
      for type <- [
            {:u, 8},
            {:u, 16},
            {:u, 32},
            {:u, 64},
            {:s, 8},
            {:s, 16},
            {:s, 32},
            {:s, 64},
            {:f, 16},
            {:f, 32},
            {:bf, 16},
            {:c, 64},
            {:pred, 8}
          ] do
        assert Mlx.Dtype.supported?(type), "Expected #{inspect(type)} to be supported"
      end
    end

    test "unsupported types" do
      refute Mlx.Dtype.supported?({:f, 64})
      refute Mlx.Dtype.supported?({:c, 128})
    end
  end

  # ── Mlx.Compiler container traversal ──

  describe "Mlx.Compiler with map outputs" do
    test "defn returning a map" do
      fun =
        Nx.Defn.jit(
          fn x ->
            %{sum: Nx.sum(x), max: Nx.reduce_max(x)}
          end,
          compiler: Mlx.Compiler
        )

      result = fun.(Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_number(result.sum) == 6.0
      assert Nx.to_number(result.max) == 3.0
    end
  end

  describe "Mlx.Compiler with list outputs" do
    test "defn returning a list" do
      fun =
        Nx.Defn.jit(
          fn x ->
            [Nx.add(x, 1), Nx.multiply(x, 2)]
          end,
          compiler: Mlx.Compiler
        )

      [a, b] = fun.(Nx.tensor([1.0, 2.0]))
      assert Nx.to_flat_list(a) == [2.0, 3.0]
      assert Nx.to_flat_list(b) == [2.0, 4.0]
    end
  end

  describe "Mlx.Compiler with nested containers" do
    test "defn returning nested tuple/map" do
      fun =
        Nx.Defn.jit(
          fn x ->
            {Nx.sum(x), %{doubled: Nx.multiply(x, 2)}}
          end,
          compiler: Mlx.Compiler
        )

      {sum, map} = fun.(Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_number(sum) == 6.0
      assert Nx.to_flat_list(map.doubled) == [2.0, 4.0, 6.0]
    end
  end

  describe "Mlx.Compiler with scalar outputs" do
    test "defn returning scalar tensor" do
      fun =
        Nx.Defn.jit(
          fn x ->
            Nx.sum(x)
          end,
          compiler: Mlx.Compiler
        )

      result = fun.(Nx.tensor([10.0, 20.0, 30.0]))
      assert Nx.to_number(result) == 60.0
    end
  end

  describe "Mlx.Compiler with non-tensor passthrough" do
    test "defn with integer operations" do
      fun =
        Nx.Defn.jit(
          fn x ->
            Nx.reshape(x, {2, 2})
          end,
          compiler: Mlx.Compiler
        )

      result = fun.(Nx.tensor([1.0, 2.0, 3.0, 4.0]))
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0, 4.0]
    end
  end

  describe "Mlx.Compiler callbacks" do
    test "__partitions_options__ returns expected format" do
      assert [groups: [default: :ok]] = Mlx.Compiler.__partitions_options__([])
    end

    test "__to_backend__ returns Mlx.Backend" do
      assert {Mlx.Backend, []} = Mlx.Compiler.__to_backend__([])
    end
  end

  # ── Mlx.Backend: less-tested dtype paths ──

  describe "tensor creation across dtypes" do
    test "u8 tensor roundtrip" do
      t = Nx.tensor([0, 127, 255], type: :u8, backend: Mlx.Backend)
      assert Nx.to_flat_list(t) == [0, 127, 255]
    end

    test "u16 tensor roundtrip" do
      t = Nx.tensor([0, 1000, 65535], type: :u16, backend: Mlx.Backend)
      assert Nx.to_flat_list(t) == [0, 1000, 65535]
    end

    test "s8 tensor roundtrip" do
      t = Nx.tensor([-128, 0, 127], type: :s8, backend: Mlx.Backend)
      assert Nx.to_flat_list(t) == [-128, 0, 127]
    end

    test "s16 tensor roundtrip" do
      t = Nx.tensor([-32768, 0, 32767], type: :s16, backend: Mlx.Backend)
      assert Nx.to_flat_list(t) == [-32768, 0, 32767]
    end

    test "f16 tensor roundtrip" do
      t = Nx.tensor([1.0, -2.0, 0.5], type: :f16, backend: Mlx.Backend)
      assert Nx.to_flat_list(t) == [1.0, -2.0, 0.5]
    end

    test "bf16 tensor roundtrip" do
      t = Nx.tensor([1.0, -2.0, 3.0], type: :bf16, backend: Mlx.Backend)
      assert Nx.to_flat_list(t) == [1.0, -2.0, 3.0]
    end

    test "c64 tensor creation" do
      t =
        Nx.tensor([Complex.new(1.0, 2.0), Complex.new(3.0, 4.0)],
          type: :c64,
          backend: Mlx.Backend
        )

      assert Nx.shape(t) == {2}
      assert Nx.type(t) == {:c, 64}
    end

    test "pred/bool tensor roundtrip" do
      t = Nx.tensor([1, 0, 1], type: {:u, 8}, backend: Mlx.Backend)
      result = Nx.equal(t, 1)
      assert Nx.to_flat_list(result) == [1, 0, 1]
    end
  end

  describe "conjugate" do
    test "conjugate of complex tensor" do
      t =
        Nx.tensor([Complex.new(1.0, 2.0), Complex.new(3.0, -4.0)],
          type: :c64,
          backend: Mlx.Backend
        )

      result = Nx.conjugate(t)
      list = Nx.to_flat_list(result)
      assert length(list) == 2
      [c1, c2] = list
      assert Complex.new(1.0, -2.0) == c1
      assert Complex.new(3.0, 4.0) == c2
    end
  end

  describe "sort descending" do
    test "sort descending reverses order" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
      result = Nx.sort(t, direction: :desc)
      assert Nx.to_flat_list(result) == [5.0, 4.0, 3.0, 1.0, 1.0]
    end

    test "argsort descending" do
      t = Nx.tensor([3.0, 1.0, 4.0])
      result = Nx.argsort(t, direction: :desc)
      assert Nx.to_flat_list(result) == [2, 0, 1]
    end
  end

  describe "gather (take_along_axis)" do
    test "gather with indices" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      indices = Nx.tensor([[0], [1], [0]])
      result = Nx.gather(t, indices)
      assert Nx.shape(result) == {3, 2}
    end
  end

  describe "inspect" do
    test "tensor inspect returns readable string" do
      t = Nx.tensor([1.0, 2.0, 3.0], backend: Mlx.Backend)
      str = inspect(t)
      assert str =~ "Nx.Tensor"
      assert str =~ "f32"
    end

    test "large tensor inspect is sliced" do
      t = Nx.iota({100}, type: :f32, backend: Mlx.Backend)
      str = inspect(t)
      assert str =~ "Nx.Tensor"
    end
  end
end
