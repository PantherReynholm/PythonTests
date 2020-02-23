import numpy as np
from random import random


def constant_velocity(particles, truth):

    # x, v_x, y, v_y, z, v_z, weight
    particle_storage = np.zeros((len(particles), 7))
    weights = []

    for i, particle in enumerate(particles):
        x_purtubation = random()
        y_purtubation = random()
        z_purtubation = random()

        particle[[0, 1]] += x_purtubation
        particle[[2, 3]] += y_purtubation
        particle[[4, 5]] += z_purtubation

        particle_storage[i, [0, 1]] = particle[[0, 1]]
        particle_storage[i, [2, 3]] = particle[[2, 3]]
        particle_storage[i, [4, 5]] = particle[[4, 5]]

        weights.append(1 / np.linalg.norm((particle_storage[i, [0, 2, 4]] - truth[[0, 1, 2]])))
    weights = np.array(weights)
    particle_storage[:, 6] = weights / max(weights)
    return particle_storage
