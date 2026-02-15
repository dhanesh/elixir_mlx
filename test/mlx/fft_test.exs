defmodule Mlx.FFTTest do
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-4)
    a_list = Nx.to_flat_list(a)
    b_list = Nx.to_flat_list(b)

    Enum.zip(a_list, b_list)
    |> Enum.each(fn {av, bv} ->
      assert_in_delta(av, bv, atol, "expected #{av} to be close to #{bv}")
    end)
  end

  describe "fft/ifft roundtrip" do
    test "1D FFT roundtrip preserves signal" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      freq = Mlx.FFT.fft(a)
      recovered = Mlx.FFT.ifft(freq)
      # Real parts should match original
      real = Nx.real(recovered)
      assert_all_close(real, a)
    end

    test "1D FFT of impulse gives constant spectrum" do
      a = Nx.tensor([1.0, 0.0, 0.0, 0.0])
      freq = Mlx.FFT.fft(a)
      # FFT of [1,0,0,0] = [1,1,1,1]
      real = Nx.real(freq)
      assert_all_close(real, Nx.tensor([1.0, 1.0, 1.0, 1.0]))
    end
  end

  describe "rfft/irfft" do
    test "rfft produces half-spectrum" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      freq = Mlx.FFT.rfft(a)
      # rfft of length 4 produces length 3 (n/2 + 1)
      assert elem(Nx.shape(freq), 0) == 3
    end

    test "rfft/irfft roundtrip" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      freq = Mlx.FFT.rfft(a)
      recovered = Mlx.FFT.irfft(freq)
      assert_all_close(recovered, a)
    end
  end

  describe "fft2/ifft2" do
    test "2D FFT roundtrip" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      freq = Mlx.FFT.fft2(a)
      recovered = Mlx.FFT.ifft2(freq)
      real = Nx.real(recovered)
      assert_all_close(real, a)
    end
  end

  describe "fftn/ifftn" do
    test "N-dim FFT roundtrip" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      freq = Mlx.FFT.fftn(a)
      recovered = Mlx.FFT.ifftn(freq)
      real = Nx.real(recovered)
      assert_all_close(real, a)
    end
  end

  describe "rfft2/irfft2" do
    test "2D real FFT roundtrip" do
      a = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
      freq = Mlx.FFT.rfft2(a)
      recovered = Mlx.FFT.irfft2(freq)
      assert_all_close(recovered, a)
    end
  end

  describe "rfftn/irfftn" do
    test "N-dim real FFT roundtrip" do
      a = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
      freq = Mlx.FFT.rfftn(a)
      recovered = Mlx.FFT.irfftn(freq)
      assert_all_close(recovered, a)
    end
  end

  describe "options" do
    test "fft with custom n (zero-pad)" do
      a = Nx.tensor([1.0, 2.0])
      freq = Mlx.FFT.fft(a, n: 4)
      assert elem(Nx.shape(freq), 0) == 4
    end

    test "fft with custom axis" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      freq = Mlx.FFT.fft(a, axis: 0)
      assert Nx.shape(freq) == {2, 2}
    end
  end
end
