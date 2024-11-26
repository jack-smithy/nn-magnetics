import time

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def define_movement_side():
    magnetization = (
        0,
        1000,
        0,
    )

    dimension = (5, 5, 5)

    position = (0.0, 0.0, 0.0)

    nx = 3
    ny = 6
    nz = 9
    xs = np.linspace(-10, -8, nx)
    ys = np.linspace(-2.0, 8.0, ny)
    zs = np.linspace(-8.0, 8.0, nz)

    sposition = np.array([[x, y, z] for x in xs for y in ys for z in zs] * 4)

    magnet_angles = np.array((0.0, 90.0, 180.0, 270.0))
    magnet_angles = np.repeat(magnet_angles, nx * ny * nz)

    magnet = magpy.magnet.Cuboid(
        dimension=dimension,
        magnetization=magnetization,
        position=position,
    )

    magnet.rotate_from_angax(magnet_angles, axis="z", anchor=None, start="auto")

    s1 = magpy.Sensor(sposition)

    return magnet, s1
