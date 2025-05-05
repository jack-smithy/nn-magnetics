from nn_magnetics.optimize.fit import optimize
from nn_magnetics.models import AngleAmpCorrectionNetwork
from nn_magnetics.optimize.mock import get_mock_measurements

A, B = 1, 1
SUSCEPTIBILITY = (0.2, 0.2, 0.2)
PATH = "/Users/jacksmith/Documents/work/nn-magnetics/results/3dof_chi_v2/2025-05-02 13:54:13.447989/best_weights.pt"


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
        round(
            abs(SUSCEPTIBILITY[i] - susc_mean[i].item()) / SUSCEPTIBILITY[i] * 100,
            precision,
        )
        for i in range(3)
    ]

    return f"""
    chi_x={a_mean}±{a_std}, chi_y={b_mean}±{b_std}, chi_z={c_mean}±{c_std}
    errors: x={errs[0]}%, y={errs[1]}%, z={errs[2]}%, overall: {sum(errs)/3}%
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
        n_steps=1000,
        n_repeats=2,
        verbose=True,
    )

    print(format_results(susc_mean, susc_std))


if __name__ == "__main__":
    main()
