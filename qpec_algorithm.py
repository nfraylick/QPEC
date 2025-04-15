import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from qiskit import Aer

# Simulated PWM Data
X_raw = np.load("pwm_dataset.npy")  # shape: (n_samples, 4)
Y = np.load("pwm_labels.npy")       # shape: (n_samples,)

# Step 1: Preprocess (scale)
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

# Step 2: Quantum Circuit Setup
n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits, shots=1024)

def feature_map(x):
    for i in range(n_qubits):
        qml.RY(np.pi * x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

def variational_block(params):
    for i in range(n_qubits):
        qml.RZ(params[i], wires=i)
        qml.RX(params[i + n_qubits], wires=i)

@qml.qnode(dev)
def quantum_circuit(x, params):
    feature_map(x)
    variational_block(params)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Step 3: Quantum Feature Extraction
def extract_features(X, params):
    return np.array([quantum_circuit(x, params) for x in X])

# Step 4: Train Variational Classifier
params = np.random.uniform(0, 2 * np.pi, size=(2 * n_qubits,), requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

for epoch in range(20):
    def cost(params_):
        X_feats = extract_features(X_train, params_)
        clf = SVC(kernel='linear').fit(X_feats, y_train)
        return 1 - clf.score(X_feats, y_train)

    params = opt.step(cost, params)
    print(f"Epoch {epoch}, Cost: {cost(params)}")

# Step 5: Final Evaluation
X_train_features = extract_features(X_train, params)
X_test_features = extract_features(X_test, params)

final_model = SVC(kernel='rbf')
final_model.fit(X_train_features, y_train)
y_pred = final_model.predict(X_test_features)

print(classification_report(y_test, y_pred))
