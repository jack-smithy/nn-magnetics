import json
import datetime
import os
from wakepy import keep

from nn_magnetics.optimize import fit, prepare_measurements, result_to_dict, evaluate

SAVE_DIR = f"fits/{datetime.datetime.now()}"
# SAVE_DIR = "fits/2024-12-11 16:26:56.382257"


def main():
    polarization_magnitude = 1.2003

    (
        positions1,
        positions2_rotated,
        field_measured1,
        field_measured2_rotated,
    ) = prepare_measurements()

    os.makedirs(f"{SAVE_DIR}")

    result = fit(
        polarization_magnitude=polarization_magnitude,
        positions1=positions1,
        positions2_rotated=positions2_rotated,
        field_measured1=field_measured1,
        field_measured2_rotated=field_measured2_rotated,
        maxiter=1000,
        popsize=40,
        save_path=SAVE_DIR,
    )

    results_dict = result_to_dict(result)

    with open(f"{SAVE_DIR}/optimised.json", "w+") as f:
        json.dump(results_dict, f)

    # with open(f"{SAVE_DIR}/best_params.json", "r") as f:
    #     result_dict = json.load(f)

    evaluate(
        result.x,
        positions1=positions1,
        positions2_rotated=positions2_rotated,
        field_measured1=field_measured1,
        field_measured2_rotated=field_measured2_rotated,
        save_dir=SAVE_DIR,
    )


if __name__ == "__main__":
    with keep.running():
        main()
