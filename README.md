# Federated IDS for IoT Edge Devices – Python Prototype

## Main Python file
`federated_ids.py`

## What this prototype demonstrates
- Four virtual IoT edge clients with separate local traffic datasets.
- Non-IID data distributions (clients have different attack proportions).
- Local logistic-regression training on each client.
- Federated Averaging (FedAvg) at a simulated central server.
- A global model evaluated after every federated round.
- Centralized baseline comparison.
- Accuracy, precision, recall, F1-score and false-positive rate.
- Estimated model-update communication cost.
- CSV, text report and PNG graphs.

## Important scope note
This is a classroom simulation using generated synthetic traffic, not CICIoT2023 or live network packets. It demonstrates the federated-learning workflow. It does not implement cryptographic secure aggregation, differential privacy, Flower/gRPC networking, or production-grade intrusion detection.

## Requirements
- Python 3.9 or newer
- NumPy
- Matplotlib

## Setup on macOS (VS Code)

1. Extract the ZIP file.
2. Open the extracted `federated_ids_prototype` folder in VS Code.
3. Open Terminal in VS Code: **Terminal → New Terminal**.
4. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

5. Activate it:

   ```bash
   source .venv/bin/activate
   ```

6. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

7. Run the prototype:

   ```bash
   python federated_ids.py
   ```

8. Open the `results` folder. It will be created automatically and contain:
   - `federated_round_results.csv`
   - `final_report.txt`
   - `federated_accuracy.png`
   - `model_comparison.png`

## How to explain the prototype
Each virtual IoT client keeps its own generated traffic records and trains a local model. It sends only its model weights and bias to the simulated aggregation server. The server combines those updates using sample-count-weighted FedAvg and broadcasts the new global model in the next round. The final model predicts whether unseen traffic is normal or an attack.

## Main file name to save
`federated_ids.py`
