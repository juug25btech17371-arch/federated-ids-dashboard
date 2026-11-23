"""
Federated IDS for IoT Edge Devices
A beginner-friendly simulation of four IoT clients and a FedAvg server.

Run:
    python federated_ids.py

The program creates synthetic traffic locally for each simulated client.
Raw client traffic is not sent to the aggregation server; only model weights
and sample counts are used for federated averaging.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
CLIENT_COUNT = 4
ROUNDS = 12
LOCAL_EPOCHS = 35
LEARNING_RATE = 0.08
SAMPLES_PER_CLIENT = 500

FEATURE_NAMES = [
    "packet_rate", "byte_rate", "connection_count", "failed_connections",
    "destination_ports", "packet_size", "duration", "syn_ratio"
]


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def make_client_data(client_id, samples, rng):
    """Create a different (non-IID) synthetic traffic shard for each client."""
    attack_ratio = [0.20, 0.35, 0.50, 0.65][client_id]
    y = (rng.random(samples) < attack_ratio).astype(float)

    # Normal traffic and attack traffic have different statistical patterns.
    normal = rng.normal(
        loc=[2.0, 2.2, 2.0, 0.5, 2.0, 2.5, 3.0, 0.15],
        scale=[0.7, 0.8, 0.7, 0.4, 0.8, 0.8, 1.0, 0.08],
        size=(samples, 8)
    )
    attack = rng.normal(
        loc=[5.0, 5.2, 4.8, 3.8, 4.5, 4.2, 1.3, 0.85],
        scale=[1.0, 1.1, 1.0, 1.0, 1.0, 1.0, 0.7, 0.12],
        size=(samples, 8)
    )

    X = np.where(y[:, None] == 1, attack, normal)

    # Small client-specific shift emulates heterogeneous edge devices.
    X += client_id * np.array([0.05, -0.03, 0.04, 0.02, 0.03, 0.01, 0.02, 0.0])
    X = np.clip(X, 0, None)
    return X.astype(float), y.astype(float)


def train_local_model(X, y, weights, bias, epochs=LOCAL_EPOCHS):
    """Train a logistic-regression model locally using gradient descent."""
    w = weights.copy()
    b = float(bias)
    n = len(y)

    for _ in range(epochs):
        predictions = sigmoid(X @ w + b)
        error = predictions - y
        grad_w = (X.T @ error) / n
        grad_b = float(np.mean(error))
        w -= LEARNING_RATE * grad_w
        b -= LEARNING_RATE * grad_b

    return w, b


def predict(X, weights, bias):
    return (sigmoid(X @ weights + bias) >= 0.5).astype(int)


def calculate_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def save_round_results(rows):
    path = OUTPUT_DIR / "federated_round_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(round_rows, final_metrics, centralized_metrics):
    rounds = [row["round"] for row in round_rows]
    accuracies = [row["accuracy"] for row in round_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracies, marker="o", label="Federated global model")
    plt.xlabel("Federated training round")
    plt.ylabel("Accuracy")
    plt.title("Federated IDS: Accuracy Across Training Rounds")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "federated_accuracy.png", dpi=150)
    plt.close()

    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    fed_values = [final_metrics[k.lower() if k != "F1-score" else "f1_score"]
                  for k in ["Accuracy", "Precision", "Recall", "F1-score"]]
    cen_values = [centralized_metrics[k.lower() if k != "F1-score" else "f1_score"]
                  for k in ["Accuracy", "Precision", "Recall", "F1-score"]]

    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, fed_values, width, label="Federated")
    plt.bar(x + width/2, cen_values, width, label="Centralized")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Federated vs Centralized IDS")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close()


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    print("=" * 62)
    print(" FEDERATED INTRUSION DETECTION SYSTEM – IoT EDGE SIMULATION")
    print("=" * 62)

    clients = []
    for client_id in range(CLIENT_COUNT):
        X, y = make_client_data(client_id, SAMPLES_PER_CLIENT, rng)
        clients.append((X, y))
        print(f"IoT Client {client_id + 1}: {len(y)} local traffic records "
              f"({int(y.sum())} attacks, {int((y == 0).sum())} normal)")

    # A separate synthetic test set is used for consistent evaluation.
    X_test, y_test = make_client_data(99, 800, rng)

    global_weights = np.zeros(len(FEATURE_NAMES), dtype=float)
    global_bias = 0.0
    round_rows = []

    print("\nStarting federated training with FedAvg...")
    for round_number in range(1, ROUNDS + 1):
        client_weights = []
        client_biases = []
        client_sizes = []

        for client_id, (X_local, y_local) in enumerate(clients, start=1):
            local_w, local_b = train_local_model(
                X_local, y_local, global_weights, global_bias
            )
            client_weights.append(local_w)
            client_biases.append(local_b)
            client_sizes.append(len(y_local))

        # FedAvg: weighted average of client model parameters.
        total_samples = sum(client_sizes)
        global_weights = sum(
            w * (size / total_samples)
            for w, size in zip(client_weights, client_sizes)
        )
        global_bias = sum(
            b * (size / total_samples)
            for b, size in zip(client_biases, client_sizes)
        )

        predictions = predict(X_test, global_weights, global_bias)
        metrics = calculate_metrics(y_test.astype(int), predictions)
        round_rows.append({
            "round": round_number,
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1_score"], 4)
        })
        print(f"Round {round_number:02d}/{ROUNDS} | "
              f"Accuracy: {metrics['accuracy']:.3f} | "
              f"F1: {metrics['f1_score']:.3f}")

    # Centralized baseline: same training records pooled for comparison.
    X_all = np.vstack([client[0] for client in clients])
    y_all = np.concatenate([client[1] for client in clients])
    central_w = np.zeros(len(FEATURE_NAMES), dtype=float)
    central_b = 0.0
    central_w, central_b = train_local_model(
        X_all, y_all, central_w, central_b, epochs=ROUNDS * LOCAL_EPOCHS
    )

    final_predictions = predict(X_test, global_weights, global_bias)
    final_metrics = calculate_metrics(y_test.astype(int), final_predictions)
    central_predictions = predict(X_test, central_w, central_b)
    centralized_metrics = calculate_metrics(y_test.astype(int), central_predictions)

    save_round_results(round_rows)
    plot_results(round_rows, final_metrics, centralized_metrics)

    # Count transmitted model parameters (weights + bias), not raw traffic.
    parameter_count = len(FEATURE_NAMES) + 1
    upload_bytes_per_round = CLIENT_COUNT * parameter_count * 8
    total_upload_bytes = upload_bytes_per_round * ROUNDS

    with (OUTPUT_DIR / "final_report.txt").open("w", encoding="utf-8") as f:
        f.write("FEDERATED IDS – SIMULATION REPORT\n")
        f.write("=" * 38 + "\n")
        f.write("Dataset: Synthetic IoT network traffic (not a real-world benchmark)\n")
        f.write(f"Clients: {CLIENT_COUNT}\nRounds: {ROUNDS}\n")
        f.write(f"Records per client: {SAMPLES_PER_CLIENT}\n\n")
        f.write("Federated model metrics:\n")
        for key, value in final_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\nCentralized baseline metrics:\n")
        for key, value in centralized_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\nCommunication estimate:\n")
        f.write(f"Model parameters per client: {parameter_count}\n")
        f.write(f"Client-to-server upload per round: {upload_bytes_per_round} bytes\n")
        f.write(f"Total client-to-server upload: {total_upload_bytes} bytes\n")
        f.write("\nPrivacy note: Raw traffic remains in each simulated client process. ")
        f.write("This educational simulation does not implement cryptographic secure aggregation.\n")

    print("\n" + "=" * 62)
    print("FINAL FEDERATED MODEL")
    for key in ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]:
        print(f"{key.replace('_', ' ').title():24s}: {final_metrics[key]:.4f}")
    print("\nCENTRALIZED BASELINE")
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        print(f"{key.replace('_', ' ').title():24s}: {centralized_metrics[key]:.4f}")
    print(f"\nEstimated client upload per round: {upload_bytes_per_round} bytes")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")
    print("Note: this is a synthetic educational simulation, not a deployed IDS.")


if __name__ == "__main__":
    main()
