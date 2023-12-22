from functools import lru_cache

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="localhost:8500")


def call_triton_model(image: np.ndarray):
    triton_client = get_client()

    input_image = InferInput(
        name="IMAGES", shape=list(image.shape), datatype=np_to_triton_dtype(image.dtype)
    )
    input_image.set_data_from_numpy(image, binary_data=True)

    infer_output = InferRequestedOutput("LOGITS", binary_data=False)
    query_response = triton_client.infer(
        model_name="cnn_onnx", inputs=[input_image], outputs=[infer_output]
    )
    logits = query_response.as_numpy("LOGITS")
    return logits


def main():
    image = np.zeros((1, 1, 28, 28), dtype=np.float32)
    logits = call_triton_model(image)
    logits_true = [
        -2.0692716,
        -2.0118053,
        -0.5460156,
        -0.42701477,
        -1.7434562,
        -0.660521,
        -0.87440807,
        -1.9680347,
        -0.14993061,
        -1.2688434,
    ]
    logits_true = np.array(logits_true)
    assert np.abs(logits - logits_true).all() < 1e6, "Something wents wrong"


if __name__ == "__main__":
    main()
