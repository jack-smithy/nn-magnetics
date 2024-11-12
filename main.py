import datetime

import wandb
from nn_magnetics import run


def main():
    EPOCHS = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    SAVE_PATH = f"results/{str(datetime.datetime.now())}"
    DATA_DIR = "data/isotropic_chi"

    wandb.init(
        # set the wandb project where this run will be logged
        project="isotropic-chi",
        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "MLP-SiLU-single-width",
            "loss": "correction-loss",
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
        save_path=SAVE_PATH,
        log=True,
    )


if __name__ == "__main__":
    main()
