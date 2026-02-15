defmodule Mlx.LinalgTest do
  use ExUnit.Case, async: true

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

  describe "inv" do
    test "inverts a 2x2 matrix" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      inv = Mlx.Linalg.inv(a)
      # A @ A^-1 should be identity
      identity = Nx.dot(a, inv)
      assert_all_close(identity, Nx.tensor([[1.0, 0.0], [0.0, 1.0]]))
    end
  end

  describe "pinv" do
    test "pseudo-inverse of a matrix" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      pinv = Mlx.Linalg.pinv(a)
      # A @ pinv(A) @ A ≈ A
      result = a |> Nx.dot(pinv) |> Nx.dot(a)
      assert_all_close(result, a)
    end
  end

  describe "cholesky" do
    test "decomposes a positive definite matrix" do
      # Create a positive definite matrix: A^T @ A
      a = Nx.tensor([[2.0, 1.0], [1.0, 3.0]])
      l = Mlx.Linalg.cholesky(a)
      # L @ L^T should equal A
      reconstructed = Nx.dot(l, Nx.transpose(l))
      assert_all_close(reconstructed, a)
    end
  end

  describe "qr" do
    test "decomposes a matrix into Q and R" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      {q, r} = Mlx.Linalg.qr(a)
      # Q @ R should equal A
      reconstructed = Nx.dot(q, r)
      assert_all_close(reconstructed, a)
    end

    test "Q is orthogonal" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      {q, _r} = Mlx.Linalg.qr(a)
      # Q^T @ Q should be identity
      identity = Nx.dot(Nx.transpose(q), q)
      assert_all_close(identity, Nx.tensor([[1.0, 0.0], [0.0, 1.0]]))
    end
  end

  describe "svd" do
    test "decomposes a matrix into U, S, Vt" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      {u, s_vals, vt} = Mlx.Linalg.svd(a)
      assert tuple_size(Nx.shape(u)) == 2
      assert tuple_size(Nx.shape(s_vals)) == 1
      assert tuple_size(Nx.shape(vt)) == 2
    end
  end

  describe "eigh" do
    test "eigendecomposition of symmetric matrix" do
      a = Nx.tensor([[2.0, 1.0], [1.0, 3.0]])
      {eigenvalues, eigenvectors} = Mlx.Linalg.eigh(a)
      assert Nx.shape(eigenvalues) == {2}
      assert Nx.shape(eigenvectors) == {2, 2}
    end
  end

  describe "eigvalsh" do
    test "eigenvalues of symmetric matrix" do
      a = Nx.tensor([[2.0, 1.0], [1.0, 3.0]])
      eigenvalues = Mlx.Linalg.eigvalsh(a)
      assert Nx.shape(eigenvalues) == {2}
      # Eigenvalues of [[2,1],[1,3]] are (5±√5)/2
      vals = Nx.to_flat_list(eigenvalues)
      assert_in_delta(Enum.min(vals), 1.381966, 1.0e-4)
      assert_in_delta(Enum.max(vals), 3.618034, 1.0e-4)
    end
  end

  describe "solve" do
    test "solves linear system Ax = b" do
      a = Nx.tensor([[3.0, 1.0], [1.0, 2.0]])
      b = Nx.tensor([[9.0], [8.0]])
      x = Mlx.Linalg.solve(a, b)
      # A @ x should equal b
      result = Nx.dot(a, x)
      assert_all_close(result, b)
    end
  end

  describe "solve_triangular" do
    test "solves lower triangular system" do
      a = Nx.tensor([[2.0, 0.0], [1.0, 3.0]])
      b = Nx.tensor([[4.0], [7.0]])
      x = Mlx.Linalg.solve_triangular(a, b, upper: false)
      result = Nx.dot(a, x)
      assert_all_close(result, b)
    end
  end

  describe "cross" do
    test "computes cross product of 3D vectors" do
      a = Nx.tensor([1.0, 0.0, 0.0])
      b = Nx.tensor([0.0, 1.0, 0.0])
      result = Mlx.Linalg.cross(a, b)
      assert_all_close(result, Nx.tensor([0.0, 0.0, 1.0]))
    end
  end

  describe "norm" do
    test "computes Frobenius norm of matrix" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      n = Mlx.Linalg.norm(a)
      # sqrt(1 + 4 + 9 + 16) = sqrt(30)
      expected = :math.sqrt(30.0)
      assert_in_delta(Nx.to_number(n), expected, 1.0e-4)
    end

    test "computes norm along axis" do
      a = Nx.tensor([[3.0, 4.0], [5.0, 12.0]])
      n = Mlx.Linalg.norm(a, axes: [1])
      assert_all_close(n, Nx.tensor([5.0, 13.0]))
    end
  end

  describe "lu" do
    test "LU decomposition returns 3 components" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      {p, l, u} = Mlx.Linalg.lu(a)
      assert tuple_size(Nx.shape(p)) >= 1
      assert tuple_size(Nx.shape(l)) == 2
      assert tuple_size(Nx.shape(u)) == 2
    end
  end

  describe "tri_inv" do
    test "inverts a lower triangular matrix" do
      a = Nx.tensor([[2.0, 0.0], [1.0, 3.0]])
      inv = Mlx.Linalg.tri_inv(a, upper: false)
      identity = Nx.dot(a, inv)
      assert_all_close(identity, Nx.tensor([[1.0, 0.0], [0.0, 1.0]]))
    end
  end

  # --- Nx.LinAlg integration tests (via Nx.Backend callbacks) ---

  describe "Nx.LinAlg.triangular_solve/3" do
    test "solves lower triangular system via Nx" do
      a = Nx.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
      b = Nx.tensor([1.0, 2.0, 1.0])
      x = Nx.LinAlg.triangular_solve(a, b)
      result = Nx.dot(a, x)
      assert_all_close(result, b)
    end

    test "solves upper triangular system via Nx" do
      a = Nx.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
      b = Nx.tensor([3.0, 2.0, 1.0])
      x = Nx.LinAlg.triangular_solve(a, b, lower: false)
      result = Nx.dot(a, x)
      assert_all_close(result, b)
    end
  end

  describe "Nx.LinAlg.qr/1" do
    test "QR decomposition via Nx" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      {q, r} = Nx.LinAlg.qr(a)
      reconstructed = Nx.dot(q, r)
      assert_all_close(reconstructed, a)
    end
  end

  describe "Nx.LinAlg.cholesky/1" do
    test "Cholesky decomposition via Nx" do
      a = Nx.tensor([[4.0, 2.0], [2.0, 3.0]])
      l = Nx.LinAlg.cholesky(a)
      reconstructed = Nx.dot(l, Nx.transpose(l))
      assert_all_close(reconstructed, a)
    end
  end

  describe "Nx.LinAlg.lu/1" do
    test "LU decomposition via Nx" do
      a = Nx.tensor([[2.0, 1.0], [4.0, 3.0]])
      {p, l, u} = Nx.LinAlg.lu(a)
      assert Nx.shape(p) == {2, 2}
      assert Nx.shape(l) == {2, 2}
      assert Nx.shape(u) == {2, 2}
      # P @ L @ U should equal A
      reconstructed = p |> Nx.dot(l) |> Nx.dot(u)
      assert_all_close(reconstructed, a)
    end
  end

  describe "Nx.LinAlg.svd/1" do
    test "SVD via Nx" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      {u, s_vals, vt} = Nx.LinAlg.svd(a)
      assert Nx.shape(u) == {2, 2}
      assert Nx.shape(s_vals) == {2}
      assert Nx.shape(vt) == {2, 2}
    end
  end

  describe "Nx.LinAlg.solve/2" do
    test "solves Ax = b via Nx" do
      a = Nx.tensor([[1.0, 3.0], [2.0, 1.0]])
      b = Nx.tensor([9.0, 8.0])
      x = Nx.LinAlg.solve(a, b)
      result = Nx.dot(a, x)
      assert_all_close(result, b)
    end
  end

  describe "Nx.LinAlg.eigh/1" do
    test "eigendecomposition via Nx" do
      a = Nx.tensor([[2.0, 1.0], [1.0, 3.0]])
      {eigenvalues, eigenvectors} = Nx.LinAlg.eigh(a)
      assert Nx.shape(eigenvalues) == {2}
      assert Nx.shape(eigenvectors) == {2, 2}
    end
  end
end
