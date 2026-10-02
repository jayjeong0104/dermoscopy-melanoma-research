"""Convergence-checked logistic heads with the original Keras L2 convention."""
from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from artifact_io import atomic_output


def objective_gradient(theta, x, y, l2):
    weights, bias = theta[:-1], theta[-1]
    logits = x @ weights + bias
    residual = expit(logits) - y
    loss = np.mean(np.logaddexp(0., logits) - y * logits) + l2 * (weights @ weights)
    gradient = np.r_[x.T @ residual / len(y) + 2 * l2 * weights, residual.mean()]
    return float(loss), gradient


def standardized(x, mean, scale):
    return np.ascontiguousarray((np.asarray(x, dtype=np.float64) - mean) / scale)


def probabilities(parameters, x):
    return expit(standardized(x, parameters["mean"], parameters["scale"]) @
                 parameters["weights"] + parameters["bias"])


def fit(x, y, l2, seed=0, callback=None, maxiter=2000):
    x, y = np.asarray(x), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or y.shape != (len(x),) or set(np.unique(y)) != {0, 1}
            or not np.isfinite(x).all() or not np.isfinite(l2) or l2 <= 0):
        raise ValueError("Need finite aligned features, both binary classes, and positive L2.")
    mean = x.mean(axis=0, dtype=np.float64).astype("float32")
    variance = np.maximum(x.var(axis=0, dtype=np.float64), 1e-6).astype("float32")
    scale = np.sqrt(variance.astype("float64"))
    z = standardized(x, mean, scale)
    initial = np.zeros(x.shape[1] + 1)
    if seed:
        initial[:-1] = np.random.default_rng(seed).normal(0, .01, x.shape[1])
    initial[-1] = np.log(y.mean() / (1 - y.mean()))
    result = minimize(objective_gradient, initial, args=(z, y, l2), jac=True,
        method="L-BFGS-B", callback=callback,
        options={"maxiter": maxiter, "maxfun": 10000, "maxls": 50, "maxcor": 20,
                 "gtol": 1e-8, "ftol": 1e-15})
    loss, gradient = objective_gradient(result.x, z, y, l2)
    norm = float(np.max(np.abs(gradient)))
    if not result.success or not np.isfinite(loss) or norm > 1e-6:
        raise RuntimeError(f"Classifier did not converge: {result.message}; max gradient={norm:g}")
    parameters = {"mean": mean, "scale": scale, "weights": result.x[:-1],
                  "bias": np.asarray(result.x[-1])}
    return parameters, {"objective": loss, "gradient_max_abs": norm,
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "solver_message": str(result.message), "converged": True, "l2": l2, "seed": seed}


def save_parameters(path: Path, parameters):
    with atomic_output(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez(stream, **parameters)


def load_parameters(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        parameters = {name: saved[name] for name in ("mean", "scale", "weights", "bias")}
    dimensions = len(parameters["weights"])
    if (any(parameters[key].shape != (dimensions,) for key in ("mean", "scale", "weights"))
            or parameters["bias"].shape != () or (parameters["scale"] <= 0).any()
            or not all(np.isfinite(value).all() for value in parameters.values())):
        raise ValueError("Invalid classifier parameters.")
    return parameters


def audit_keras_head(path: Path, x, y):
    # Read only the known linear-head format, without deserializing executable objects.
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
        layers = config["config"]["layers"]
        if [layer["class_name"] for layer in layers] != ["InputLayer", "Normalization", "Dense"]:
            raise ValueError("Only the archived standardized logistic head is supported.")
        norm, dense = layers[1]["config"], layers[2]["config"]
        if dense["activation"] != "sigmoid" or dense["units"] != 1 or norm["invert"]:
            raise ValueError("Unexpected archived head configuration.")
        l2 = float(dense["kernel_regularizer"]["config"]["l2"])
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as weights:
            theta = np.r_[weights["layers/dense/vars/0"][:].ravel(), weights["layers/dense/vars/1"][:]]
    mean, variance = np.asarray(norm["mean"]), np.asarray(norm["variance"])
    np.testing.assert_allclose(mean, x.mean(axis=0, dtype=np.float64), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(variance, np.maximum(x.var(axis=0, dtype=np.float64), 1e-6), rtol=1e-5, atol=1e-6)
    loss, gradient = objective_gradient(theta, standardized(x, mean, np.sqrt(variance)), y, l2)
    return {"objective_float64": loss, "gradient_max_abs": float(np.abs(gradient).max()),
            "l2": l2, "scope": "Full-training objective at saved Adam weights, evaluated in float64."}
