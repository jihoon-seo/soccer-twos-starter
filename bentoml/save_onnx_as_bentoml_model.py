import bentoml
import onnx

onnx_model = onnx.load_model("./SoccerTwos.onnx")

result = bentoml.onnx.save_model("soccer_twos_onnx", onnx_model)
print(result)
