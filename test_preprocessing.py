def test_scaled_shape():
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    X = np.random.rand(100, 4) * 100
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == (100, 4), "Shape mismatch"
    assert np.all((X_scaled >= 0) & (X_scaled <= 1)), "Scaling bounds error"

test_scaled_shape()
print("test_preprocessing.py passed.")