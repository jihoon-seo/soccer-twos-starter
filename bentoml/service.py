import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# You need to save a pre-trained model to local BentoML store with the name `soccer_twos_onnx` in advance.

bentoml_onnx_model: bentoml.Model = bentoml.onnx.get("soccer_twos_onnx")

bentoml_onnx_runner = bentoml_onnx_model.to_runner()

svc = bentoml.Service("soccer_twos_onnx_predictor", runners=[bentoml_onnx_runner])

ort_session = bentoml.onnx.load_model("soccer_twos_onnx")

action_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
action_mask = np.vstack((action_mask)).transpose()

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:
    X_test1 = np.vstack((input_series)).transpose()

    result = ort_session.run(["discrete_actions"], {
             "vector_observation": X_test1.astype(np.float32),
             "action_masks": action_mask.astype(np.float32),
             })[0] # 'result' contains "discrete_actions" output

    return np.array((np.argmax(result[0][:3]), np.argmax(result[0][3:6]), np.argmax(result[0][6:9]))) # result[0] == agent_1
    