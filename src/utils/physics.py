import numpy as np


def demagnetizing_factor(a, b, c):
    norm = np.sqrt(a * a + b * b + c * c)
    normab = np.sqrt(a * a + b * b)
    normac = np.sqrt(a * a + c * c)
    normbc = np.sqrt(b * b + c * c)

    return (1 / np.pi) * (
        (b**2 - c**2) / (2 * b * c) * np.log((norm - a) / (norm + a))
        + (a**2 - c**2) / (2 * a * c) * np.log((norm - b) / (norm + b))
        + b / (2 * c) * np.log((normab + a) / (normab - a))
        + a / (2 * c) * np.log((normab + b) / (normab - b))
        + c / (2 * a) * np.log((normbc - b) / (normbc + b))
        + c / (2 * b) * np.log((normac - a) / (normac + a))
        + 2 * np.arctan((a * b) / (c * norm))
        + (a**3 + b**3 - 2 * c**3) / (3 * a * b * c)
        + (a**2 + b**2 - 2 * c**2) / (3 * a * b * c) * norm
        + c / (a * b) * (np.sqrt(a**2 + c**2) + np.sqrt(b**2 + c**2))
        - (
            (a**2 + b**2) ** (3 / 2)
            + (b**2 + c**2) ** (3 / 2)
            + (c**2 + a**2) ** (3 / 2)
        )
        / (3 * a * b * c)
    )
