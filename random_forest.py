# QPEC: Quantum Pattern Extraction and Classification

# Section 1: Preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_raw = np.load("pwm_dataset.npy")  # shape: (n_samples, 4)
y_raw = np.load("pwm_labels.npy")   # shape: (n_samples,)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=0.2)

# Section 2: Quantum Feature Encoding and Extraction
import pennylane as qml
from pennylane import numpy as pnp

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def amplitude_encoder(x):
    qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')

def variational_block(params):
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev)
def quantum_feature_circuit(x, params):
    amplitude_encoder(x)
    variational_block(params)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def extract_quantum_features(X, params):
    return np.array([quantum_feature_circuit(x, params) for x in X])

# Section 3: Hybrid Quantum-Classical Training
from sklearn.ensemble import RandomForestClassifier

params = pnp.array(np.random.uniform(0, np.pi, n_qubits), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.05)

def cost_fn(params):
    feats = extract_quantum_features(X_train, params)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(feats, y_train)
    acc = model.score(feats, y_train)
    return 1 - acc

# Section 4: Training Convergence and Optimization
epochs = 25
for epoch in range(epochs):
    params = opt.step(cost_fn, params)
    if epoch % 5 == 0:
        current_cost = cost_fn(params)
        print(f"Epoch {epoch}: Cost = {current_cost:.4f}")

# Section 5: Classification
X_train_q = extract_quantum_features(X_train, params)
X_test_q = extract_quantum_features(X_test, params)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_q, y_train)
y_pred = clf.predict(X_test_q)

# Section 6: Evaluation
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Section 7: Output to File
import pandas as pd

results_df = pd.DataFrame({
    'True Label': y_test,
    'Predicted': y_pred
})
results_df.to_csv("qpec_results.csv", index=False)
print("Results saved to qpec_results.csv")
