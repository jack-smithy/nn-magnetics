from nn_magnetics.optimize.fit import optimize
from nn_magnetics.models import AngleAmpCorrectionNetwork
from nn_magnetics.optimize.mock import get_mock_measurements

A, B = 1, 1
SUSCEPTIBILITY = (0.2, 0.2, 0.2)
PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/3dof_chi_v2/2025-04-30 15:39:23.157513/best_weights.pt"


def format_results(susc_mean, susc_std, precision: int = 5) -> str:
    """Process and format the results"""

    a_mean, b_mean, c_mean = (
        round(susc_mean[0].item(), precision),
        round(susc_mean[1].item(), precision),
        round(susc_mean[2].item(), precision),
    )

    a_std, b_std, c_std = (
        round(susc_std[0].item(), precision),
        round(susc_std[1].item(), precision),
        round(susc_std[2].item(), precision),
    )

    errs = [
        abs(SUSCEPTIBILITY[i] - susc_mean[i].item()) / SUSCEPTIBILITY[i] * 100
        for i in range(3)
    ]

    return f"""
    chi_x={a_mean}±{a_std}, chi_y={b_mean}±{b_std}, chi_z={c_mean}±{c_std}
    {errs}
    """


def main():
    observers, B_measured = get_mock_measurements(
        a=A,
        b=B,
        susceptibility=SUSCEPTIBILITY,
    )

    susc_mean, susc_std = optimize(
        model_cls=AngleAmpCorrectionNetwork,
        path=PATH,
        observers=observers,
        B_measured=B_measured,
        a=A,
        b=B,
        n_steps=100,
        n_repeats=20,
    )

    print(format_results(susc_mean, susc_std))


if __name__ == "__main__":
    main()
