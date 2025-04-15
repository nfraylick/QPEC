import numpy as np
import matplotlib.pyplot as plt

signal = np.load("pwm_sample.npy")
plt.plot(signal)
plt.title("Captured PWM Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (raw ADC)")
plt.grid(True)
plt.savefig("pwm_visualization.png")
plt.show()