from pynq import Overlay, allocate
from pynq.lib import AxiGPIO
import numpy as np
import time

# Load custom overlay with PWM acquisition logic
overlay = Overlay("/home/xilinx/pwm_acquisition.bit")
gpio_config = overlay.axi_gpio_0
dma = overlay.axi_dma_0

# Configure GPIO for PWM input
gpio_config.write(0, 0b0001)  # enable PWM capture mode

# Allocate buffers
buffer_size = 2048
in_buffer = allocate(shape=(buffer_size,), dtype=np.uint16)

# Configure DMA for stream
dma.recvchannel.transfer(in_buffer)
time.sleep(0.1)  # Let it collect data
dma.recvchannel.wait()

# Save the acquired signal
np.save("/home/xilinx/pwm_signal.npy", in_buffer)
print("PWM signal captured and saved.")