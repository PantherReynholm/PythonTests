import numpy as np
from random import random, randint
import pandas as pd
import os


def particle_resampler(particles):

    lower_probability_bound = 1 / len(particles)
    high_probability_particles = []
    counter = 0

    for i, particle in enumerate(particles):
        if particle[6] < (lower_probability_bound * (len(particles) / 6)):
            np.delete(particles, i, 0)
            counter += 1
        elif weights[i] > lower_probability_bound * 3:
            high_probability_particles.append(particles[i])
        else:
            pass
    print(len(particles))
    print(high_probability_particles)
    for i in range(counter):
        np.vstack((particles, high_probability_particles[randint(0, len(high_probability_particles))]))

    return particles
