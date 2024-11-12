import datetime

import wandb
from nn_magnetics import run


def main():
    EPOCHS = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    DATA_DIR = "data/isotropic_chi"

    wandb.init(
        # set the wandb project where this run will be logged
        project="anisotropic-chi",
        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "MLP-single-width",
            "activation": "SiLU",
            "loss": "field-loss",
            "dataset": "data/isotropic_chi",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
    )

    run(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        data_dir=DATA_DIR,
        save_path=f"results/{wandb.run.name if wandb.run is not None else str(datetime.datetime.now())}",
        log=True,
    )


if __name__ == "__main__":
    main()
