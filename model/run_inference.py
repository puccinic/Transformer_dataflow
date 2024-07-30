import numpy as np
import onnx
from qonnx.util.exec_qonnx import exec_qonnx

rng = np.random.default_rng()
model = onnx.load("modified_Encoder.onnx")
np.save("input.npy", rng.random(size=(1, 65, 512), dtype=np.float32) / 0.03125)
exec_qonnx("modified_Encoder.onnx", "input.npy")
