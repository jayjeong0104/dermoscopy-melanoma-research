"""Convergence-checked diagnosis softmax head for frozen image embeddings."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from converged_probe import standardized, save_parameters

CLASSES = ("AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC")


def objective_gradient(theta, x, labels, l2, classes):
    coefficients = theta.reshape(x.shape[1] + 1, classes)
    weights, bias = coefficients[:-1], coefficients[-1]
    logits = x @ weights + bias
    loss = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels])
    residual = softmax(logits, axis=1)
    residual[np.arange(len(labels)), labels] -= 1
    gradient = np.vstack((x.T @ residual / len(labels) + 2 * l2 * weights, residual.mean(axis=0)))
    return float(loss + l2 * np.sum(weights ** 2)), gradient.ravel()


def probabilities(parameters, x):
    return softmax(standardized(x, parameters["mean"], parameters["scale"]) @
                   parameters["weights"] + parameters["bias"], axis=1)


def fit(x, diagnoses, l2, seed=0, callback=None, maxiter=2000, classes=CLASSES):
    x, diagnoses = np.asarray(x), np.asarray(diagnoses)
    classes = tuple(classes)
    if (x.ndim != 2 or x.shape[1] == 0 or diagnoses.shape != (len(x),)
            or not np.isfinite(x).all() or len(set(classes)) != len(classes)
            or len(classes) < 2 or set(diagnoses) != set(classes)
            or not np.isfinite(l2) or l2 <= 0):
        raise ValueError("Need finite aligned features, every declared diagnosis, and positive L2.")
    y = np.array([classes.index(value) for value in diagnoses], dtype=int)
    mean = x.mean(axis=0, dtype=np.float64).astype("float32")
    scale = np.sqrt(np.maximum(x.var(axis=0, dtype=np.float64), 1e-6).astype("float32").astype("float64"))
    z = standardized(x, mean, scale)
    initial = np.zeros((x.shape[1] + 1, len(classes)))
    if seed:
        initial[:-1] = np.random.default_rng(seed).normal(0, .01, initial[:-1].shape)
    initial[-1] = np.log(np.bincount(y, minlength=len(classes)) / len(y))
    initial[-1] -= initial[-1].mean()
    result = minimize(objective_gradient, initial.ravel(), args=(z, y, l2, len(classes)),
        method="L-BFGS-B", jac=True, callback=callback,
        options={"maxiter": maxiter, "maxfun": 15000, "maxls": 50, "maxcor": 20,
                 "gtol": 1e-8, "ftol": 1e-15})
    loss, gradient = objective_gradient(result.x, z, y, l2, len(classes))
    norm = float(np.max(np.abs(gradient)))
    # SciPy's stopping target is stricter than the declared first-order
    # acceptance criterion. A finite convex fit can reach the iteration limit
    # after satisfying that criterion; independently verify it instead of
    # treating the solver's iteration budget as evidence of a bad optimum.
    # Abnormal line searches and function-evaluation limits still fail closed.
    iteration_limit = (result.status == 1 and
        "ITERATIONS REACHED LIMIT" in str(result.message).upper())
    first_order_optimal = (np.isfinite(result.x).all() and np.isfinite(loss)
        and np.isfinite(gradient).all() and norm <= 1e-6)
    if not first_order_optimal or not (result.success or iteration_limit):
        raise RuntimeError(f"Diagnosis head did not converge: {result.message}; max gradient={norm:g}")
    coefficients = result.x.reshape(initial.shape)
    parameters = {"mean": mean, "scale": scale, "weights": coefficients[:-1],
        "bias": coefficients[-1] - coefficients[-1].mean(), "classes": np.asarray(classes)}
    return parameters, {"objective": loss, "gradient_max_abs": norm, "iterations": int(result.nit),
        "function_evaluations": int(result.nfev), "solver_success": bool(result.success),
        "solver_status": int(result.status), "solver_message": str(result.message),
        "convergence_criterion": "finite objective and coefficients; max absolute analytic gradient <= 1e-6",
        "termination_acceptance": "solver_success" if result.success else "first_order_optimality_at_iteration_limit",
        "converged": True, "l2": l2, "seed": seed}


def load_parameters(path):
    with np.load(path, allow_pickle=False) as saved:
        result = {name: saved[name] for name in ("mean", "scale", "weights", "bias", "classes")}
    if result["weights"].ndim != 2:
        raise ValueError("Invalid diagnosis weight dimensions.")
    dimensions, count = result["weights"].shape
    if (result["mean"].shape != (dimensions,) or result["scale"].shape != (dimensions,)
            or result["bias"].shape != (count,) or result["classes"].shape != (count,)
            or len(set(result["classes"])) != count or "MEL" not in result["classes"]
            or (result["scale"] <= 0).any()
            or not all(np.isfinite(result[name]).all() for name in ("mean", "scale", "weights", "bias"))):
        raise ValueError("Invalid diagnosis parameters or class mapping.")
    return result


def export(parameters_path, output_path, verification_features):
    import tensorflow as tf
    from artifact_io import save_model
    from compare_model_runs import file_hash
    from foundation_common import save_json
    parameters = load_parameters(parameters_path)
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input((len(parameters["mean"]),), dtype="float64", name="frozen_image_embedding")
    normalized = tf.keras.layers.Normalization(mean=parameters["mean"].astype("float64"),
        variance=parameters["scale"] ** 2, dtype="float64")(inputs)
    dense = tf.keras.layers.Dense(len(parameters["classes"]), activation="softmax", dtype="float64")
    diagnoses = dense(normalized)
    melanoma = tf.keras.layers.Dense(1, use_bias=False, trainable=False, dtype="float64")
    model = tf.keras.Model(inputs, {"diagnosis_probabilities": diagnoses, "melanoma_probability": melanoma(diagnoses)})
    dense.set_weights([parameters["weights"], parameters["bias"]])
    selector = (parameters["classes"] == "MEL").astype("float64")[:, None]
    melanoma.set_weights([selector])
    save_model(model, output_path)
    restored = tf.keras.models.load_model(output_path, compile=False)
    expected = probabilities(parameters, verification_features)
    actual = restored(np.asarray(verification_features, dtype="float64"), training=False)
    np.testing.assert_allclose(actual["diagnosis_probabilities"], expected, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(actual["melanoma_probability"], expected @ selector, rtol=1e-7, atol=1e-8)
    report = {"tensorflow": tf.__version__, "classes": parameters["classes"].tolist(),
        "source_parameters_sha256": file_hash(parameters_path), "checkpoint_sha256": file_hash(output_path),
        "maximum_probability_difference": float(np.max(np.abs(actual["diagnosis_probabilities"].numpy() - expected))),
        "verified_rows": len(expected), "input": "Frozen lesion embeddings, not JPEG images",
        "outputs_are_not_clinically_calibrated_diagnoses": True, "decision_threshold_included": False,
        "automatic_promotion": False}
    save_json(output_path.with_suffix(".export.json"), report)
    return report
