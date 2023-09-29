FCN_CONFIG = {
    "input_dim": 784,
    "output_dim": 10,
    "hidden_dim_1": 256,
    "hidden_dim_2": 128,
    "dropout": 0.3,
}
CNN_CONFIG = {"output_dim": 10, "dropout": 0.5}
FCN_TRAIN_CONFIG = {"batch_size": 256, "n_epochs": 10}
CNN_TRAIN_CONFIG = {"batch_size": 256, "n_epochs": 10}
FCN_OPTIMIZER_CONFIG = {"lr": 1e-3}
CNN_OPTIMIZER_CONFIG = {"lr": 1e-3}
