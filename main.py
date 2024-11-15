import datetime

import wandb
from nn_magnetics import train, evaluate


def main():
    project = "anisotropic_chi_test"
    name = wandb.run.name if wandb.run is not None else str(datetime.datetime.now())

    config = {
        "learning_rate": 0.004,
        "architecture": "MLP",
        "hidden_dim_factor": 6,
        "activation": "SiLU",
        "loss_fn": "field",
        "data_dir": "data/anisotropic_chi",
        "epochs": 10,
        "batch_size": 256,
        "gamma": 0.9,
        "save_path": f"results/{project}/{name}",
    }

    if False:
        wandb.init(
            project=project,
            config=config,
        )

    train(config)
    evaluate(config)


if __name__ == "__main__":
    main()
