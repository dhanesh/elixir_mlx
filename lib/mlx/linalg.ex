defmodule Mlx.Linalg do
  @moduledoc """
  Linear algebra operations using MLX.

  Provides matrix decompositions, solvers, and norm computations
  accelerated on Apple Silicon via Metal.

  ## Example

      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Mlx.Backend)
      {q, r} = Mlx.Linalg.qr(a)
  """

  alias Mlx.NIF

  defp s, do: elem(NIF.default_cpu_stream(), 1)

  defp unwrap!({:ok, val}), do: val
  defp unwrap!({:error, msg}), do: raise("MLX error: #{msg}")

  defp from_ref(%Nx.Tensor{data: %Mlx.Backend{ref: ref}}), do: ref

  @doc """
  Computes the inverse of a square matrix.
  """
  def inv(%Nx.Tensor{} = a) do
    ref = unwrap!(NIF.mlx_linalg_inv(from_ref(a), s()))
    to_nx(a, ref)
  end

  @doc """
  Computes the pseudo-inverse of a matrix.
  """
  def pinv(%Nx.Tensor{} = a) do
    ref = unwrap!(NIF.mlx_linalg_pinv(from_ref(a), s()))
    to_nx(a, ref)
  end

  @doc """
  Computes the Cholesky decomposition.

  ## Options
    * `:upper` - if true, return upper triangular (default `false`)
  """
  def cholesky(%Nx.Tensor{} = a, opts \\ []) do
    upper = if Keyword.get(opts, :upper, false), do: true, else: false
    ref = unwrap!(NIF.mlx_linalg_cholesky(from_ref(a), upper, s()))
    to_nx(a, ref)
  end

  @doc """
  Computes the inverse from a Cholesky decomposition.

  ## Options
    * `:upper` - if true, input is upper triangular (default `false`)
  """
  def cholesky_inv(%Nx.Tensor{} = a, opts \\ []) do
    upper = if Keyword.get(opts, :upper, false), do: true, else: false
    ref = unwrap!(NIF.mlx_linalg_cholesky_inv(from_ref(a), upper, s()))
    to_nx(a, ref)
  end

  @doc """
  Computes the QR decomposition.

  Returns `{q, r}` where `a = q @ r`.
  """
  def qr(%Nx.Tensor{} = a) do
    {q_ref, r_ref} = unwrap!(NIF.mlx_linalg_qr(from_ref(a), s()))
    {to_nx_infer(q_ref, a.type), to_nx_infer(r_ref, a.type)}
  end

  @doc """
  Computes the Singular Value Decomposition.

  Returns `{u, s, vt}` where `a = u @ diag(s) @ vt`.
  """
  def svd(%Nx.Tensor{} = a) do
    parts = unwrap!(NIF.mlx_linalg_svd(from_ref(a), s()))

    Enum.map(parts, &to_nx_infer(&1, a.type))
    |> List.to_tuple()
  end

  @doc """
  Computes eigenvalues and eigenvectors of a symmetric/Hermitian matrix.

  Returns `{eigenvalues, eigenvectors}`.

  ## Options
    * `:uplo` - `:L` for lower (default) or `:U` for upper triangular input
  """
  def eigh(%Nx.Tensor{} = a, opts \\ []) do
    uplo = Keyword.get(opts, :uplo, :L)
    {ev_ref, evec_ref} = unwrap!(NIF.mlx_linalg_eigh(from_ref(a), uplo, s()))
    {to_nx_infer(ev_ref, a.type), to_nx_infer(evec_ref, a.type)}
  end

  @doc """
  Computes eigenvalues of a symmetric/Hermitian matrix.

  ## Options
    * `:uplo` - `:L` for lower (default) or `:U` for upper triangular input
  """
  def eigvalsh(%Nx.Tensor{} = a, opts \\ []) do
    uplo = Keyword.get(opts, :uplo, :L)
    ref = unwrap!(NIF.mlx_linalg_eigvalsh(from_ref(a), uplo, s()))
    to_nx_infer(ref, a.type)
  end

  @doc """
  Computes the LU decomposition.

  Returns `{p, l, u}` (permutation, lower, upper).
  """
  def lu(%Nx.Tensor{} = a) do
    parts = unwrap!(NIF.mlx_linalg_lu(from_ref(a), s()))

    Enum.map(parts, &to_nx_infer(&1, a.type))
    |> List.to_tuple()
  end

  @doc """
  Computes the LU factorization (compact form).

  Returns `{lu, pivots}`.
  """
  def lu_factor(%Nx.Tensor{} = a) do
    {lu_ref, piv_ref} = unwrap!(NIF.mlx_linalg_lu_factor(from_ref(a), s()))
    {to_nx_infer(lu_ref, a.type), to_nx_infer(piv_ref, {:s, 32})}
  end

  @doc """
  Solves the linear system `a @ x = b`.
  """
  def solve(%Nx.Tensor{} = a, %Nx.Tensor{} = b) do
    ref = unwrap!(NIF.mlx_linalg_solve(from_ref(a), from_ref(b), s()))
    to_nx_infer(ref, a.type)
  end

  @doc """
  Solves a triangular linear system `a @ x = b`.

  ## Options
    * `:upper` - if true, `a` is upper triangular (default `false`)
  """
  def solve_triangular(%Nx.Tensor{} = a, %Nx.Tensor{} = b, opts \\ []) do
    upper = if Keyword.get(opts, :upper, false), do: true, else: false
    ref = unwrap!(NIF.mlx_linalg_solve_triangular(from_ref(a), from_ref(b), upper, s()))
    to_nx_infer(ref, a.type)
  end

  @doc """
  Computes the cross product of two 3-element vectors.

  ## Options
    * `:axis` - axis along which to compute (default `-1`)
  """
  def cross(%Nx.Tensor{} = a, %Nx.Tensor{} = b, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)
    ref = unwrap!(NIF.mlx_linalg_cross(from_ref(a), from_ref(b), axis, s()))
    to_nx(a, ref)
  end

  @doc """
  Computes the inverse of a triangular matrix.

  ## Options
    * `:upper` - if true, input is upper triangular (default `false`)
  """
  def tri_inv(%Nx.Tensor{} = a, opts \\ []) do
    upper = if Keyword.get(opts, :upper, false), do: true, else: false
    ref = unwrap!(NIF.mlx_linalg_tri_inv(from_ref(a), upper, s()))
    to_nx(a, ref)
  end

  @doc """
  Computes the matrix or vector norm.

  ## Options
    * `:axes` - axes along which to compute (default: all)
    * `:keepdims` - keep reduced dimensions (default `false`)
  """
  def norm(%Nx.Tensor{} = a, opts \\ []) do
    axes = Keyword.get(opts, :axes, nil)
    keepdims = if Keyword.get(opts, :keepdims, false), do: true, else: false
    ref = unwrap!(NIF.mlx_linalg_norm(from_ref(a), axes, keepdims, s()))
    to_nx_infer(ref, a.type)
  end

  @doc """
  Computes the matrix or vector norm with specified order.

  ## Options
    * `:axes` - axes along which to compute (default: all)
    * `:keepdims` - keep reduced dimensions (default `false`)
  """
  def norm(%Nx.Tensor{} = a, ord, opts) when is_number(ord) do
    axes = Keyword.get(opts, :axes, nil)
    keepdims = if Keyword.get(opts, :keepdims, false), do: true, else: false
    ref = unwrap!(NIF.mlx_linalg_norm_p(from_ref(a), ord / 1, axes, keepdims, s()))
    to_nx_infer(ref, a.type)
  end

  # Helpers

  defp to_nx(template, ref) do
    %{template | data: %Mlx.Backend{ref: ref}}
  end

  defp to_nx_infer(ref, default_type) do
    {:ok, shape_list} = NIF.shape(ref)
    {:ok, dtype_atom} = NIF.dtype(ref)

    type =
      case Mlx.Dtype.to_nx(dtype_atom) do
        nil -> default_type
        t -> t
      end

    shape = List.to_tuple(shape_list)

    %Nx.Tensor{
      data: %Mlx.Backend{ref: ref},
      type: type,
      shape: shape,
      names: List.duplicate(nil, tuple_size(shape))
    }
  end
end
