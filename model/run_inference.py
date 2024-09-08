import os
import numpy as np
import torch
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.cleanup import cleanup_model

onnx_model = ModelWrapper("./vit_tiny_quant.onnx")
cleanup_model(onnx_model)
inferred_model = onnx_model.transform(InferShapes())
inferred_model = inferred_model.transform(InferDataTypes())
np_images = np.random.rand(1,65,512)
with torch.no_grad():
    outputs = execute_onnx(
        inferred_model,
        {'/Add_1_output_0': np_images},
        return_full_exec_context=True,
        start_node="/transformer/layers.0.0/norm/BatchNormalization",
        end_node="/transformer/Add_1"
    )


import numpy as np
import onnx
import onnx.utils
from onnx import checker
from onnx import numpy_helper
from onnxruntime import InferenceSession


ort_sess = InferenceSession("vit_bn_RELU.onnx")
inputs = ort_sess.get_inputs()
outputs = ort_sess.get_outputs()
output_names = [output.name for output in outputs]
rng = np.random.default_rng()
ort_dict = dict()


print("Inputs: \n", inputs)
print("Outputs: \n", outputs)

for input in inputs:
  ort_dict[input.name] = rng.random(size=tuple(input.shape), dtype=np.float32)

pred_onx = ort_sess.run(output_names, ort_dict)

print("Predictions:\n", pred_onx)