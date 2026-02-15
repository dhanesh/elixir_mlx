#!/usr/bin/env python3
"""
Generate reference values from Python MLX for numerical comparison tests.

Usage:
    pip install mlx
    python test/scripts/generate_references.py

Outputs: test/fixtures/python_refs/references.json
"""

import json
import math
import mlx.core as mx
import mlx.nn as nn


def to_list(arr):
    """Convert MLX array to nested Python list."""
    return arr.tolist()


def main():
    refs = {}

    # ── Basic unary ops ──────────────────────────────────
    x = mx.array([0.5, 1.0, 2.0, 3.0, 4.0])

    refs["exp"] = {"input": to_list(x), "output": to_list(mx.exp(x))}
    refs["log"] = {"input": to_list(x), "output": to_list(mx.log(x))}
    refs["sqrt"] = {"input": to_list(x), "output": to_list(mx.sqrt(x))}
    refs["rsqrt"] = {"input": to_list(x), "output": to_list(mx.rsqrt(x))}
    refs["abs"] = {"input": to_list(-x), "output": to_list(mx.abs(-x))}
    refs["negative"] = {"input": to_list(x), "output": to_list(-x)}
    refs["square"] = {"input": to_list(x), "output": to_list(mx.square(x))}

    # Trig ops
    t = mx.array([0.0, 0.5, 1.0])
    refs["sin"] = {"input": to_list(t), "output": to_list(mx.sin(t))}
    refs["cos"] = {"input": to_list(t), "output": to_list(mx.cos(t))}
    refs["tanh"] = {"input": to_list(t), "output": to_list(mx.tanh(t))}

    # Special functions
    refs["erf"] = {"input": to_list(t), "output": to_list(mx.erf(t))}
    refs["sigmoid"] = {"input": to_list(mx.array([-2.0, 0.0, 2.0])),
                       "output": to_list(mx.sigmoid(mx.array([-2.0, 0.0, 2.0])))}

    # ── Binary ops ──────────────────────────────────────
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])

    refs["add"] = {"left": to_list(a), "right": to_list(b),
                   "output": to_list(a + b)}
    refs["multiply"] = {"left": to_list(a), "right": to_list(b),
                        "output": to_list(a * b)}
    refs["subtract"] = {"left": to_list(a), "right": to_list(b),
                        "output": to_list(a - b)}
    refs["divide"] = {"left": to_list(a), "right": to_list(b),
                      "output": to_list(a / b)}
    refs["power"] = {"left": to_list(a), "right": to_list(mx.array([2.0, 2.0, 2.0])),
                     "output": to_list(mx.power(a, mx.array([2.0, 2.0, 2.0])))}

    # ── Reduction ops ───────────────────────────────────
    m = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    refs["sum_all"] = {"input": to_list(m), "output": float(mx.sum(m).item())}
    refs["sum_axis0"] = {"input": to_list(m), "output": to_list(mx.sum(m, axis=0))}
    refs["sum_axis1"] = {"input": to_list(m), "output": to_list(mx.sum(m, axis=1))}
    refs["mean_all"] = {"input": to_list(m), "output": float(mx.mean(m).item())}
    refs["max_all"] = {"input": to_list(m), "output": float(mx.max(m).item())}
    refs["min_all"] = {"input": to_list(m), "output": float(mx.min(m).item())}
    refs["argmax_axis1"] = {"input": to_list(m),
                            "output": to_list(mx.argmax(m, axis=1))}

    # ── Softmax ─────────────────────────────────────────
    logits = mx.array([[1.0, 2.0, 3.0]])
    refs["softmax"] = {"input": to_list(logits),
                       "output": to_list(mx.softmax(logits, axis=-1))}

    # ── Activations (nn module) ─────────────────────────
    act_x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    refs["gelu"] = {"input": to_list(act_x), "output": to_list(nn.gelu(act_x))}
    refs["silu"] = {"input": to_list(act_x), "output": to_list(nn.silu(act_x))}
    refs["relu"] = {"input": to_list(act_x), "output": to_list(nn.relu(act_x))}
    refs["relu6"] = {"input": to_list(mx.array([-1.0, 0.0, 3.0, 6.0, 10.0])),
                     "output": to_list(nn.relu6(mx.array([-1.0, 0.0, 3.0, 6.0, 10.0])))}
    refs["elu"] = {"input": to_list(act_x), "output": to_list(nn.elu(act_x))}
    refs["softplus"] = {"input": to_list(act_x), "output": to_list(nn.softplus(act_x))}
    refs["celu"] = {"input": to_list(act_x), "output": to_list(nn.celu(act_x))}
    refs["log_sigmoid"] = {"input": to_list(act_x),
                           "output": to_list(nn.log_sigmoid(act_x))}
    refs["hard_swish"] = {"input": to_list(act_x),
                          "output": to_list(nn.hardswish(act_x))}
    refs["mish"] = {"input": to_list(act_x), "output": to_list(nn.mish(act_x))}
    refs["leaky_relu"] = {"input": to_list(act_x),
                          "output": to_list(nn.leaky_relu(act_x))}
    refs["selu"] = {"input": to_list(act_x), "output": to_list(nn.selu(act_x))}

    # ── Matmul ──────────────────────────────────────────
    A = mx.array([[1.0, 2.0], [3.0, 4.0]])
    B = mx.array([[5.0, 6.0], [7.0, 8.0]])
    refs["matmul"] = {"left": to_list(A), "right": to_list(B),
                      "output": to_list(A @ B)}

    # ── Linalg (CPU stream required for many ops) ──────
    cpu = mx.cpu
    sym = mx.array([[4.0, 2.0], [2.0, 3.0]])
    eigenvalues = mx.linalg.eigvalsh(sym, stream=cpu)
    refs["eigvalsh"] = {"input": to_list(sym),
                        "output": to_list(eigenvalues)}

    # SVD
    svd_u, svd_s, svd_vt = mx.linalg.svd(A, stream=cpu)
    refs["svd_s"] = {"input": to_list(A), "output": to_list(svd_s)}

    # Inverse
    inv_A = mx.linalg.inv(A, stream=cpu)
    refs["inv"] = {"input": to_list(A), "output": to_list(inv_A)}

    # Norm
    refs["norm"] = {"input": to_list(a), "output": float(mx.linalg.norm(a).item())}

    # ── Cumulative ops ──────────────────────────────────
    c = mx.array([1.0, 2.0, 3.0, 4.0])
    refs["cumsum"] = {"input": to_list(c), "output": to_list(mx.cumsum(c))}
    refs["cumprod"] = {"input": to_list(c), "output": to_list(mx.cumprod(c))}

    # Write output
    output_path = "test/fixtures/python_refs/references.json"
    with open(output_path, "w") as f:
        json.dump(refs, f, indent=2)

    print(f"Generated {len(refs)} reference values to {output_path}")


if __name__ == "__main__":
    main()
