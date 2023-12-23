from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
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
    image = np.load("image_example.npy")
    logits = call_triton_model(image)
    probs = nn.functional.softmax(torch.tensor(logits), dim=1)
    digit = torch.argmax(probs)
    assert digit == 5, "Something wents wrong"


if __name__ == "__main__":
    main()
