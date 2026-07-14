import numpy as np


def wrap_deg(angle):
    return np.mod(angle, 360.0)


def angle_diff(target, source):
    return (target - source + 180.0) % 360.0 - 180.0
