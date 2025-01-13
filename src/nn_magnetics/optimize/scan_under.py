import time

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


def define_movement_under():
    magnetization = (
        0,
        1000,
        0,
    )

    dimension = (5, 5, 5)

    position = (0.0, 0.0, 0.0)

    nx = 5
    ny = 5
    nz = 3
    xs = np.linspace(-2.0, 2.0, nx)
    ys = np.linspace(-2.0, 2.0, ny)
    zs = np.linspace(-7.5, -5.5, nz)

    sposition = np.array([[x, y, z] for x in xs for y in ys for z in zs])

    magnet = magpy.magnet.Cuboid(
        dimension=dimension, magnetization=magnetization, position=position
    )

    s1 = magpy.Sensor(sposition)

    return magnet, s1
