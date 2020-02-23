import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from preprocessing_data import lat_long_to_cartesian, create_prior
from forecaster import constant_velocity
from resampler import particle_resampler

file_list = os.listdir()
file_number = 0
number_of_particles = 100

file_path = file_list[file_number]
df = pd.read_csv(file_path)
df = np.array(df)
df = df[5000: 7000, 1:5]
df = np.array(df, dtype=float)

location = lat_long_to_cartesian(df)

prior = create_prior(location)

initial_particles = np.random.multivariate_normal(prior, np.diag([1, 1, 1, 1, 0.01, 1]), number_of_particles)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(location[:, 0], location[:, 1], location[:, 2])
ax.scatter(
        [particles[0] for particles in initial_particles],
        [particles[2] for particles in initial_particles],
        [particles[4] for particles in initial_particles],
        color='coral')
# plt.show()

dynamic_model = constant_velocity(initial_particles, location[0])
particle_resampler(dynamic_model)

