defmodule Mlx.NN do
  @moduledoc """
  Functional neural network building blocks for MLX.

  This module provides stateless activation functions, loss functions, and
  functional layer operations that compose Nx operations backed by MLX.
  These are the building blocks — for stateful layers with parameter management,
  use Axon.

  ## Activations

      x = Nx.tensor([-1.0, 0.0, 1.0], backend: Mlx.Backend)
      Mlx.NN.relu(x)     #=> #Nx.Tensor<f32[3] [0.0, 0.0, 1.0]>
      Mlx.NN.gelu(x)     #=> approximate Gaussian error linear unit
      Mlx.NN.silu(x)     #=> x * sigmoid(x), aka swish

  ## Loss Functions

      predictions = Nx.tensor([0.9, 0.1], backend: Mlx.Backend)
      targets = Nx.tensor([1.0, 0.0], backend: Mlx.Backend)
      Mlx.NN.mse_loss(predictions, targets)

  ## Functional Layers

      # Linear transform: x @ weight^T + bias
      Mlx.NN.linear(x, weight, bias)

      # Embedding lookup
      Mlx.NN.embedding(weight_table, indices)
  """

  import Nx.Defn

  # ── Activations ──────────────────────────────────────────────

  @doc """
  Rectified Linear Unit: `max(0, x)`.
  """
  defn relu(x) do
    Nx.max(x, 0)
  end

  @doc """
  Capped ReLU: `clip(x, 0, 6)`.
  """
  defn relu6(x) do
    Nx.clip(x, 0, 6)
  end

  @doc """
  Leaky ReLU: `x` if `x > 0`, else `alpha * x`.

  Default `alpha` is 0.01.
  """
  defn leaky_relu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 0.01)
    alpha = opts[:alpha]
    Nx.select(Nx.greater(x, 0), x, Nx.multiply(x, alpha))
  end

  @doc """
  Exponential Linear Unit.

  `x` if `x > 0`, else `alpha * (exp(x) - 1)`.
  Default `alpha` is 1.0.
  """
  defn elu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0)
    alpha = opts[:alpha]
    Nx.select(Nx.greater(x, 0), x, Nx.multiply(alpha, Nx.subtract(Nx.exp(x), 1)))
  end

  @doc """
  Scaled Exponential Linear Unit.

  SELU uses fixed constants `alpha=1.6733` and `lambda=1.0507` for
  self-normalizing neural networks.
  """
  defn selu(x) do
    alpha = 1.6732632423543772
    lambda = 1.0507009873554805

    Nx.multiply(
      lambda,
      Nx.select(Nx.greater(x, 0), x, Nx.multiply(alpha, Nx.subtract(Nx.exp(x), 1)))
    )
  end

  @doc """
  Gaussian Error Linear Unit.

  `gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`.
  Uses the exact formulation via `erf`.
  """
  defn gelu(x) do
    Nx.multiply(
      x,
      Nx.multiply(0.5, Nx.add(1.0, Nx.erf(Nx.multiply(x, Nx.rsqrt(Nx.tensor(2.0))))))
    )
  end

  @doc """
  Sigmoid Linear Unit (SiLU), also known as Swish.

  `silu(x) = x * sigmoid(x)`.
  """
  defn silu(x) do
    Nx.multiply(x, Nx.sigmoid(x))
  end

  @doc "Alias for `silu/1`."
  defn(swish(x), do: silu(x))

  @doc """
  Mish activation: `x * tanh(softplus(x))`.
  """
  defn mish(x) do
    Nx.multiply(x, Nx.tanh(Nx.log(Nx.add(1.0, Nx.exp(x)))))
  end

  @doc """
  Softplus: `log(1 + exp(beta * x)) / beta`.

  Default `beta` is 1.0.
  """
  defn softplus(x, opts \\ []) do
    opts = keyword!(opts, beta: 1.0)
    beta = opts[:beta]
    Nx.divide(Nx.log(Nx.add(1.0, Nx.exp(Nx.multiply(beta, x)))), beta)
  end

  @doc """
  Continuously Differentiable ELU.

  `celu(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1))`.
  Default `alpha` is 1.0.
  """
  defn celu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0)
    alpha = opts[:alpha]

    Nx.add(
      Nx.max(x, 0),
      Nx.min(0, Nx.multiply(alpha, Nx.subtract(Nx.exp(Nx.divide(x, alpha)), 1)))
    )
  end

  @doc """
  Log-sigmoid: `-softplus(-x)` for numerical stability.
  """
  defn log_sigmoid(x) do
    Nx.negate(Nx.log(Nx.add(1.0, Nx.exp(Nx.negate(x)))))
  end

  @doc """
  Hard sigmoid: `clip(x/6 + 0.5, 0, 1)`.

  A piecewise linear approximation of sigmoid.
  """
  defn hard_sigmoid(x) do
    Nx.clip(Nx.add(Nx.divide(x, 6.0), 0.5), 0, 1)
  end

  @doc """
  Hard swish: `x * hard_sigmoid(x)`.
  """
  defn hard_swish(x) do
    Nx.multiply(x, hard_sigmoid(x))
  end

  # ── Loss Functions ──────────────────────────────────────────

  @doc """
  Mean squared error loss.

  `mean((predictions - targets)^2)`
  """
  defn mse_loss(predictions, targets) do
    diff = Nx.subtract(predictions, targets)
    Nx.mean(Nx.multiply(diff, diff))
  end

  @doc """
  Mean absolute error (L1) loss.

  `mean(|predictions - targets|)`
  """
  defn l1_loss(predictions, targets) do
    Nx.mean(Nx.abs(Nx.subtract(predictions, targets)))
  end

  @doc """
  Huber loss (smooth L1).

  Quadratic for small errors, linear for large errors.
  Default `delta` is 1.0.
  """
  defn huber_loss(predictions, targets, opts \\ []) do
    opts = keyword!(opts, delta: 1.0)
    delta = opts[:delta]
    diff = Nx.abs(Nx.subtract(predictions, targets))
    quadratic = Nx.multiply(0.5, Nx.multiply(diff, diff))
    linear = Nx.subtract(Nx.multiply(delta, diff), Nx.multiply(0.5, Nx.multiply(delta, delta)))
    Nx.mean(Nx.select(Nx.less_equal(diff, delta), quadratic, linear))
  end

  @doc """
  Binary cross-entropy loss.

  Expects `predictions` in (0, 1) and `targets` in {0, 1}.

  `BCE = -mean(targets * log(pred) + (1 - targets) * log(1 - pred))`
  """
  defn binary_cross_entropy(predictions, targets) do
    # Clamp predictions for numerical stability
    eps = 1.0e-7
    pred = Nx.clip(predictions, eps, 1.0 - eps)

    loss =
      Nx.negate(
        Nx.add(
          Nx.multiply(targets, Nx.log(pred)),
          Nx.multiply(Nx.subtract(1.0, targets), Nx.log(Nx.subtract(1.0, pred)))
        )
      )

    Nx.mean(loss)
  end

  @doc """
  Cross-entropy loss from logits.

  Computes `-sum(targets * log_softmax(logits))` over the last axis,
  then takes the mean over all other dimensions.

  ## Options
    * `:label_smoothing` - smoothing factor in [0, 1) (default: 0.0)
  """
  defn cross_entropy(logits, targets, opts \\ []) do
    opts = keyword!(opts, label_smoothing: 0.0)
    smoothing = opts[:label_smoothing]

    # log-softmax for numerical stability: logits - logsumexp(logits)
    max_logit = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_logit)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
    log_probs = Nx.subtract(shifted, log_sum_exp)

    # Apply label smoothing if requested
    num_classes = Nx.axis_size(logits, -1)

    smoothed_targets =
      Nx.add(
        Nx.multiply(Nx.subtract(1.0, smoothing), targets),
        Nx.divide(smoothing, num_classes)
      )

    # Cross-entropy: -sum(targets * log_probs, axis=-1)
    per_sample = Nx.negate(Nx.sum(Nx.multiply(smoothed_targets, log_probs), axes: [-1]))
    Nx.mean(per_sample)
  end

  # ── Functional Layers ──────────────────────────────────────

  @doc """
  Linear (fully connected) layer: `x @ weight^T + bias`.

  ## Arguments
    * `x` - input tensor of shape `{..., in_features}`
    * `weight` - weight matrix of shape `{out_features, in_features}`
    * `bias` - optional bias vector of shape `{out_features}` (default: `nil`)
  """
  defn linear(x, weight, bias \\ Nx.tensor(0.0)) do
    # x @ weight^T
    result = Nx.dot(x, [-1], weight, [-1])

    # Check if bias is a scalar 0.0 (our sentinel for "no bias")
    # Since we're in defn, we add it unconditionally — a zero bias is a no-op
    Nx.add(result, bias)
  end

  @doc """
  Embedding lookup: selects rows from a weight table by indices.

  ## Arguments
    * `weight` - embedding table of shape `{vocab_size, embed_dim}`
    * `indices` - integer indices tensor
  """
  deftransform embedding(weight, indices) do
    Nx.take(weight, indices, axis: 0)
  end

  @doc """
  Dropout: randomly zeros elements during training.

  During training, each element is zeroed with probability `rate` and
  the remaining elements are scaled by `1 / (1 - rate)`.

  ## Options
    * `:rate` - dropout probability (default: 0.5)
    * `:training` - whether in training mode (default: `true`)
    * `:key` - optional PRNG key for reproducibility
  """
  deftransform dropout(x, opts \\ []) do
    rate = Keyword.get(opts, :rate, 0.5)
    training = Keyword.get(opts, :training, true)
    key = Keyword.get(opts, :key, nil)

    if training and rate > 0.0 do
      random_opts = [shape: Nx.shape(x)]
      random_opts = if key, do: Keyword.put(random_opts, :key, key), else: random_opts
      mask = Nx.greater(Mlx.Random.uniform(random_opts), rate)
      x |> Nx.select(mask, 0) |> Nx.divide(1.0 - rate)
    else
      x
    end
  end

  # ── Re-exports (convenience delegation to Mlx.Fast) ──────

  @doc "Delegates to `Mlx.Fast.layer_norm/4`."
  defdelegate layer_norm(x, weight \\ nil, bias \\ nil, opts \\ []), to: Mlx.Fast

  @doc "Delegates to `Mlx.Fast.rms_norm/3`."
  defdelegate rms_norm(x, weight, opts \\ []), to: Mlx.Fast

  @doc "Delegates to `Mlx.Fast.scaled_dot_product_attention/5`."
  defdelegate scaled_dot_product_attention(q, k, v, scale, opts \\ []), to: Mlx.Fast

  @doc "Delegates to `Mlx.Fast.rope/2`."
  defdelegate rope(x, opts \\ []), to: Mlx.Fast
end
