# QPEC: Classical Pattern Extraction and Classification with SVM

# Section 1: Preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_raw = np.load("pwm_dataset.npy")  # shape: (n_samples, 4)
y_raw = np.load("pwm_labels.npy")   # shape: (n_samples,)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=0.2)

# Section 2: Classification with Support Vector Machine (SVM)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Section 3: Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Section 4: Output to File
results_df = pd.DataFrame({
    'True Label': y_test,
    'Predicted': y_pred
})
results_df.to_csv("qpec_results.csv", index=False)
print("Results saved to qpec_results.csv")
