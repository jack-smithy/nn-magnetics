from matplotlib import colors as C


# CMAP_ANGLE = C.LinearSegmentedColormap.from_list(
#     "Random gradient 7907",
#     (
#         # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%207907=FFFFFF-FFB629-D76B00-AA0002
#         (0.000, (1.000, 1.000, 1.000)),
#         (0.333, (1.000, 0.714, 0.161)),
#         (0.667, (0.843, 0.420, 0.000)),
#         (1.000, (0.667, 0.000, 0.008)),
#     ),
# )

CMAP_ANGLE = C.LinearSegmentedColormap.from_list(
    "Random gradient 7907",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%207907=0:FFFFFF-7.5:FFFA90-20:FFD800-34.2:FF9E00-50:FF7600-62.5:E15300-75:E10000-87.5:C60002-100:550026
        (0.000, (1.000, 1.000, 1.000)),
        (0.075, (1.000, 0.980, 0.565)),
        (0.200, (1.000, 0.847, 0.000)),
        (0.342, (1.000, 0.620, 0.000)),
        (0.500, (1.000, 0.463, 0.000)),
        (0.625, (0.882, 0.325, 0.000)),
        (0.750, (0.882, 0.000, 0.000)),
        (0.875, (0.776, 0.000, 0.008)),
        (1.000, (0.333, 0.000, 0.149)),
    ),
)
# CMAP_AMPLITUDE = C.LinearSegmentedColormap.from_list(
#     "Random gradient 7907",
#     (
#         # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%207907=600061-4C36FF-FFFFFF-FFC930-AA0002
#         (0.000, (0.376, 0.000, 0.380)),
#         (0.250, (0.298, 0.212, 1.000)),
#         (0.500, (1.000, 1.000, 1.000)),
#         (0.750, (1.000, 0.788, 0.188)),
#         (1.000, (0.667, 0.000, 0.008)),
#     ),
# )

CMAP_AMPLITUDE = C.LinearSegmentedColormap.from_list(
    "Random gradient 7907",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%207907=0:500051-25:1C38FF-40:00EAFF-50:FFFFFF-60:FFFD00-75:FFCF47-100:AA0002
        (0.000, (0.314, 0.000, 0.318)),
        (0.250, (0.110, 0.220, 1.000)),
        (0.400, (0.000, 0.918, 1.000)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.600, (1.000, 0.992, 0.000)),
        (0.750, (1.000, 0.812, 0.278)),
        (1.000, (0.667, 0.000, 0.008)),
    ),
)
