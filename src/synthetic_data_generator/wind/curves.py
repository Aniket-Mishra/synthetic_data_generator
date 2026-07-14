import numpy as np

POWER_CURVE = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (3.5, 8),
    (4, 35),
    (4.5, 77),
    (5, 137),
    (5.5, 217),
    (6, 321),
    (6.5, 450),
    (7, 607),
    (7.5, 793),
    (8, 1008),
    (8.5, 1250),
    (9, 1515),
    (9.5, 1685),
    (10, 1790),
    (10.5, 1790),
    (11, 1790),
    (11.5, 1790),
    (12, 1790),
    (12.5, 1790),
    (13, 1790),
    (13.5, 1790),
    (14, 1790),
    (14.5, 1790),
    (15, 1790),
    (15.5, 1790),
    (16, 1790),
    (16.5, 1790),
    (17, 1790),
    (17.5, 1790),
    (18, 1790),
    (18.5, 1790),
    (19, 1790),
    (19.5, 1790),
    (20, 1790),
    (20.5, 1790),
    (21, 1790),
    (21.5, 1790),
    (22, 1790),
    (22.5, 1790),
    (23, 0),
    (23.5, 0),
    (24, 0),
    (24.5, 0),
    (25, 0),
    (25.5, 0),
    (26, 0),
    (26.5, 0),
    (27, 0),
    (27.5, 0),
    (28, 0),
    (28.5, 0),
    (29, 0),
    (29.5, 0),
    (30, 0),
]

WS_POINTS = np.array([x for x, _ in POWER_CURVE], dtype=float)
P_POINTS = np.array([y for _, y in POWER_CURVE], dtype=float)


def power_curve_kw(wind_speed):
    return np.interp(wind_speed, WS_POINTS, P_POINTS)


def commanded_pitch_angle(wind_speed):
    ws = np.asarray(wind_speed, dtype=float)
    pitch = np.zeros_like(ws)

    pitch = np.where(ws < 3.0, 89.0, pitch)
    pitch = np.where((ws >= 3.0) & (ws < 10.0), 0.5 + 0.35 * (ws - 3.0), pitch)
    pitch = np.where(
        (ws >= 10.0) & (ws < 15.0), 3.0 + 1.6 * (ws - 10.0), pitch
    )
    pitch = np.where(
        (ws >= 15.0) & (ws < 20.0), 11.0 + 2.6 * (ws - 15.0), pitch
    )
    pitch = np.where(
        (ws >= 20.0) & (ws < 23.0), 24.0 + 20.0 * (ws - 20.0), pitch
    )
    pitch = np.where(ws >= 23.0, 89.0, pitch)

    return np.clip(pitch, 0.0, 90.0)
