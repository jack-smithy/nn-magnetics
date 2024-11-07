import datetime

from src import run


def main():
    EPOCHS = 5
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    SAVE_PATH = f"results/{str(datetime.datetime.now())}"
    DATA_DIR = "data/isotropic_chi_v2"

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
