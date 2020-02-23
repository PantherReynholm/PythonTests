import numpy as np
import pandas as pd
import os


def lat_long_to_cartesian(location):
    radius_of_earth = 6371000
    cartesian_coordinates = []
    for row in location:
        x = radius_of_earth * np.cos(row[1]) * np.cos(row[2])
        y = radius_of_earth * np.cos(row[1]) * np.sin(row[2])
        z = row[-1]
        t = row[0]
        cartesian_coordinates.append([x, y, z, t])
    return np.array(cartesian_coordinates)


def create_prior(input_location):
    x_0, x_1 = input_location[0, 0], input_location[2, 0]
    y_0, y_1 = input_location[0, 1], input_location[2, 1]
    z_0, z_1 = input_location[0, 2], input_location[2, 2]
    delta_t = input_location[0, -1] - input_location[2, -1]

    v_x_0 = (x_1 - x_0) / delta_t
    v_y_0 = (y_1 - y_0) / delta_t
    v_z_0 = (z_1 - z_0) / delta_t

    return np.array([x_0, v_x_0, y_0, v_y_0, z_0, v_z_0])
