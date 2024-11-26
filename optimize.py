import json

from wakepy import keep

from nn_magnetics.optimize import fit, prepare_measurements, result_to_dict, evaluate


def main():
    polarization_magnitude = 1.2003

    (
        positions1,
        positions2_rotated,
        field_measured1,
        field_measured2_rotated,
    ) = prepare_measurements()

    result = fit(
        polarization_magnitude=polarization_magnitude,
        positions1=positions1,
        positions2_rotated=positions2_rotated,
        field_measured1=field_measured1,
        field_measured2_rotated=field_measured2_rotated,
        maxiter=50,
    )

    # results_dict = result_to_dict(result)

    # with open("fits/optimised.json", "w+") as f:
    #     json.dump(results_dict, f)

    evaluate(
        result.x,
        positions1=positions1,
        positions2_rotated=positions2_rotated,
        field_measured1=field_measured1,
        field_measured2_rotated=field_measured2_rotated,
    )


if __name__ == "__main__":
    with keep.running():
        main()
