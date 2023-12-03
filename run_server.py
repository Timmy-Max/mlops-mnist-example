import json
import os
import subprocess

import hydra
import mlflow
import numpy as np
import onnx
from config import Server


@hydra.main(config_path="configs", config_name="server", version_base="1.3")
def run(cfg: Server):
    onnx_model = onnx.load_model(cfg.model_path)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    batch_size = cfg.batch_size
    inp_example = np.empty((batch_size, 1, 28, 28), dtype=np.float32)
    out_example = np.empty(batch_size, dtype=np.float32)
    signature = mlflow.models.signature.infer_signature(inp_example, out_example)
    with mlflow.start_run():
        model_info = mlflow.onnx.log_model(
            onnx_model, "onnx_model", signature=signature
        )

    model_uri = os.path.join("mlartifacts", "0", str(model_info.run_id))
    model_info_json = {
        "name": "mnist",
        "implementation": "mlserver_mlflow.MLflowRuntime",
        "parameters": {
            # "uri": model_info.model_uri,
            "uri": os.path.join(model_uri, "artifacts", "onnx_model"),
        },
    }
    with open("model-settings.json", "w") as fp:
        json.dump(model_info_json, fp, indent=2)
        fp.write("\n")

    print("Starting server")
    subprocess.run(["mlserver", "start", "."])


if __name__ == "__main__":
    run()
