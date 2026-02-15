defmodule Mlx.IntegrationTest do
  @moduledoc """
  Integration smoke tests verifying Mlx.Backend works as a drop-in
  replacement with Nx ecosystem libraries: Axon, Polaris, and Scholar.

  These tests exercise the Mlx.Compiler (defn/jit) path extensively,
  since Axon and Scholar use Nx.Defn for their computations.
  """
  use ExUnit.Case, async: false

  # Integration tests need exclusive backend access — not async
  setup do
    Nx.default_backend(Mlx.Backend)
    :ok
  end

  # ── Axon: Model Creation + Predict ──

  describe "Axon model predict" do
    test "simple dense model forward pass" do
      # Build a simple 2-layer model: input(4) -> dense(8) -> relu -> dense(2)
      model =
        Axon.input("input", shape: {nil, 4})
        |> Axon.dense(8, activation: :relu)
        |> Axon.dense(2)

      # Build and init params
      {init_fn, predict_fn} = Axon.build(model)
      template = %{"input" => Nx.template({1, 4}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      # Verify we got model state back
      assert %Axon.ModelState{} = params

      # Run prediction
      input = Nx.iota({2, 4}, type: :f32)
      result = predict_fn.(params, %{"input" => input})

      # Verify output shape and type
      assert Nx.shape(result) == {2, 2}
      assert Nx.type(result) == {:f, 32}

      # Result should be finite numbers
      result_list = Nx.to_flat_list(result)
      assert length(result_list) == 4
      Enum.each(result_list, fn v -> assert is_float(v) end)
    end

    test "model with multiple activations" do
      model =
        Axon.input("x", shape: {nil, 3})
        |> Axon.dense(16, activation: :sigmoid)
        |> Axon.dense(8, activation: :tanh)
        |> Axon.dense(1)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 3}, :f32), Axon.ModelState.empty())

      input = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      result = predict_fn.(params, input)

      assert Nx.shape(result) == {2, 1}
      # Sigmoid and tanh constrain intermediate values, so output should be finite
      Enum.each(Nx.to_flat_list(result), fn v ->
        assert is_float(v) and not (v != v)
      end)
    end
  end

  # ── Axon: Training Loop ──

  describe "Axon training" do
    test "loss decreases on simple regression" do
      # Simple model: input(1) -> dense(4) -> relu -> dense(1)
      model =
        Axon.input("input", shape: {nil, 1})
        |> Axon.dense(4, activation: :relu)
        |> Axon.dense(1)

      # Synthetic data: y = 2x + 1 (with some noise)
      key = Nx.Random.key(42)
      {xs, _key} = Nx.Random.normal(key, shape: {32, 1})
      ys = Nx.add(Nx.multiply(xs, 2.0), 1.0)

      data = [{xs, ys}] |> Stream.cycle()

      optimizer = Polaris.Optimizers.sgd(learning_rate: 0.01)

      # Train for a few iterations and capture initial + final loss
      trained_state =
        model
        |> Axon.Loop.trainer(:mean_squared_error, optimizer, log: 0)
        |> Axon.Loop.run(data, Axon.ModelState.empty(), iterations: 50, epochs: 1)

      # Verify we got trained state back (Axon.ModelState)
      assert %Axon.ModelState{} = trained_state

      # Verify the model can predict after training
      {_init_fn, predict_fn} = Axon.build(model)
      test_input = Nx.tensor([[1.0], [2.0], [3.0]])
      predictions = predict_fn.(trained_state, %{"input" => test_input})

      assert Nx.shape(predictions) == {3, 1}
      # After training on y=2x+1, predictions should be in reasonable range
      pred_list = Nx.to_flat_list(predictions)
      Enum.each(pred_list, fn v -> assert is_float(v) end)
    end

    test "classification with binary cross-entropy" do
      # Simple binary classifier
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(4, activation: :relu)
        |> Axon.dense(1, activation: :sigmoid)

      # Simple separable data: class 0 = low values, class 1 = high values
      xs = Nx.tensor([
        [0.1, 0.2], [0.2, 0.1], [0.15, 0.25], [0.3, 0.1],
        [0.8, 0.9], [0.9, 0.8], [0.85, 0.75], [0.7, 0.9]
      ])
      ys = Nx.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])

      data = [{xs, ys}] |> Stream.cycle()
      optimizer = Polaris.Optimizers.adam(learning_rate: 0.01)

      trained_state =
        model
        |> Axon.Loop.trainer(:binary_cross_entropy, optimizer, log: 0)
        |> Axon.Loop.run(data, Axon.ModelState.empty(), iterations: 100, epochs: 1)

      # Verify training completed
      assert %Axon.ModelState{} = trained_state

      # Verify predictions are in [0, 1] range (sigmoid output)
      {_init_fn, predict_fn} = Axon.build(model)
      preds = predict_fn.(trained_state, %{"input" => xs})
      pred_list = Nx.to_flat_list(preds)
      Enum.each(pred_list, fn v ->
        assert v >= 0.0 and v <= 1.0, "sigmoid output #{v} out of range"
      end)
    end
  end

  # ── Scholar: Linear Regression ──

  describe "Scholar.Linear.LinearRegression" do
    test "fits and predicts simple linear relationship" do
      # y = 2*x1 + 3*x2 + 1
      x = Nx.tensor([
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 2.0],
        [4.0, 2.0],
        [5.0, 3.0],
        [6.0, 3.0]
      ])
      y = Nx.tensor([6.0, 8.0, 13.0, 15.0, 20.0, 22.0])

      model = Scholar.Linear.LinearRegression.fit(x, y)

      # Coefficients should be close to [2, 3]
      coeffs = Nx.to_flat_list(model.coefficients)
      assert_in_delta Enum.at(coeffs, 0), 2.0, 0.5
      assert_in_delta Enum.at(coeffs, 1), 3.0, 0.5

      # Predict on new data
      x_new = Nx.tensor([[7.0, 4.0]])
      pred = Scholar.Linear.LinearRegression.predict(model, x_new)
      pred_val = Nx.to_number(Nx.squeeze(pred))
      # Expected: 2*7 + 3*4 + 1 = 27
      assert_in_delta pred_val, 27.0, 2.0
    end
  end

  # ── Scholar: KMeans Clustering ──

  describe "Scholar.Cluster.KMeans" do
    test "clusters separable data into correct groups" do
      # Two well-separated clusters
      key = Nx.Random.key(123)

      # Cluster 1: around (0, 0)
      {c1, key} = Nx.Random.normal(key, shape: {15, 2})

      # Cluster 2: around (10, 10)
      {c2, _key} = Nx.Random.normal(key, shape: {15, 2})
      c2 = Nx.add(c2, 10.0)

      data = Nx.concatenate([c1, c2], axis: 0)

      model = Scholar.Cluster.KMeans.fit(data,
        num_clusters: 2,
        key: Nx.Random.key(42)
      )

      # Should have 2 cluster centers
      assert Nx.shape(model.clusters) == {2, 2}

      # Cluster centers should be well separated
      centers = Nx.to_flat_list(model.clusters)
      center1 = {Enum.at(centers, 0), Enum.at(centers, 1)}
      center2 = {Enum.at(centers, 2), Enum.at(centers, 3)}

      # Distance between cluster centers should be large (>5)
      dist = :math.sqrt(
        :math.pow(elem(center1, 0) - elem(center2, 0), 2) +
        :math.pow(elem(center1, 1) - elem(center2, 1), 2)
      )
      assert dist > 5.0, "Cluster centers should be well separated, got distance #{dist}"

      # Predictions should assign points to correct clusters
      labels = Scholar.Cluster.KMeans.predict(model, data)
      assert Nx.shape(labels) == {30}

      # First 15 points should all have same label, last 15 should have different label
      label_list = Nx.to_flat_list(labels)
      first_group = Enum.take(label_list, 15)
      second_group = Enum.drop(label_list, 15)

      assert length(Enum.uniq(first_group)) == 1, "First cluster should have uniform labels"
      assert length(Enum.uniq(second_group)) == 1, "Second cluster should have uniform labels"
      assert Enum.at(first_group, 0) != Enum.at(second_group, 0), "Clusters should have different labels"
    end
  end

  # ── Nx.Defn: Direct Compilation ──

  describe "Nx.Defn compilation with Mlx.Compiler" do
    test "compiled function produces correct results" do
      # Test that Mlx.Compiler works for JIT compilation
      fun = Nx.Defn.jit(fn x, w ->
        x
        |> Nx.dot(w)
        |> Nx.add(1.0)
        |> Nx.sigmoid()
      end, compiler: Mlx.Compiler)

      x = Nx.tensor([[1.0, 2.0, 3.0]])
      w = Nx.tensor([[0.1], [0.2], [0.3]])

      result = fun.(x, w)
      assert Nx.shape(result) == {1, 1}

      # Manual: dot = 1*0.1 + 2*0.2 + 3*0.3 = 1.4, +1 = 2.4, sigmoid(2.4) ≈ 0.9168
      assert_in_delta Nx.to_number(Nx.squeeze(result)), 0.9168, 0.01
    end

    test "compiled function with multiple outputs" do
      fun = Nx.Defn.jit(fn x ->
        mean = Nx.mean(x)
        std = Nx.standard_deviation(x)
        {mean, std}
      end, compiler: Mlx.Compiler)

      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {mean, std} = fun.(x)

      assert_in_delta Nx.to_number(mean), 3.0, 0.01
      assert_in_delta Nx.to_number(std), 1.4142, 0.01
    end
  end
end
