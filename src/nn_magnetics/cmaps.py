from matplotlib import colors as C


CMAP_ANGLE = C.LinearSegmentedColormap.from_list(
    "Random gradient 7907",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%207907=FFFFFF-FFB629-D76B00-AA0002
        (0.000, (1.000, 1.000, 1.000)),
        (0.333, (1.000, 0.714, 0.161)),
        (0.667, (0.843, 0.420, 0.000)),
        (1.000, (0.667, 0.000, 0.008)),
    ),
)

CMAP_AMPLITUDE = C.LinearSegmentedColormap.from_list(
    "Random gradient 7907",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%207907=600061-4C36FF-FFFFFF-FFC930-AA0002
        (0.000, (0.376, 0.000, 0.380)),
        (0.250, (0.298, 0.212, 1.000)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.750, (1.000, 0.788, 0.188)),
        (1.000, (0.667, 0.000, 0.008)),
    ),
)
