from pennylane import numpy as np
import pennylane as qml

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def qc(x):
    qml.AngleEmbedding(x, wires=range(4), rotation="Y")
    qml.templates.StronglyEntanglingLayers(np.random.rand(1, 4, 3), wires=range(4))
    return qml.expval(qml.PauliZ(0))

drawer = qml.draw(qc)
print(drawer([0.1, 0.2, 0.3, 0.4]))