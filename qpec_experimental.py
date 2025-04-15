# QPEC Rev2: Experimental Hybrid Quantum-Classical PWM Classifier (Unstable Prototype)

# NOTE: This is a proof-of-concept with experimental components that are not fully tested or guaranteed to work.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pennylane as qml
from pennylane import numpy as pnp

# Experimental quantum device
try:
    dev = qml.device("lightning.qubit", wires=4)
except:
    dev = qml.device("default.qubit", wires=4)

# Load data
X_raw = np.load("pwm_dataset.npy")
y_raw = np.load("pwm_labels.npy")

X_scaled = (X_raw - X_raw.min(axis=0)) / (X_raw.max(axis=0) - X_raw.min(axis=0))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=0.2)

# Experimental feature encoder
def hybrid_encoder(x):
    qml.AngleEmbedding(x, wires=range(4), rotation="Y")
    qml.templates.StronglyEntanglingLayers(np.random.rand(1, 4, 3), wires=range(4))  # Random layer

# Random parameter generation (non-optimized)
def init_params():
    return pnp.array(np.random.rand(1, 4, 3), requires_grad=False)

params = init_params()

@qml.qnode(dev)
def quantum_features(x):
    hybrid_encoder(x)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

def extract_q_features(X):
    return np.array([quantum_features(x) for x in X])

print("Extracting experimental quantum features (this may be slow and unstable)...")
X_train_q = extract_q_features(X_train)
X_test_q = extract_q_features(X_test)

# Basic classifier (SVM replaced with logistic regression for speed)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_q, y_train)
y_pred = model.predict(X_test_q)

# Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\nExperimental QPEC Rev2 Metrics:")
print(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}")

pd.DataFrame({"True": y_test, "Pred": y_pred}).to_csv("qpec_rev2_results.csv", index=False)
print("Experimental results saved to qpec_rev2_results.csv")
