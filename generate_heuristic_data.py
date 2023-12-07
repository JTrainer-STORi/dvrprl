from dvrprl.tsp_env import TSPEnv
import numpy as np
import pandas as pd
from datetime import datetime

# Greedy heuristic

def get_distance(observation):
    return np.linalg.norm(observation[:,0:2] - observation[:,2:4], axis=1)

def act_greedily(observation):
    return np.argmin(get_distance(observation))

# Set up data generation

NUM_EPISODES = 1000

# Set up environment

NUM_NODES = 5

ENV_SEED = 186

env = TSPEnv(num_nodes=NUM_NODES, seed=ENV_SEED)

# Setup lists that will store the data

all_timesteps = [] # timestep within an episode
all_state_id = [0] # timestep across all episodes
all_distances_to_go = [] # Response variable - distance taken for the greedy heuristic to complete the episode from the corresponding state
all_x_coords = [] # x coordinate of the customer
all_y_coords = [] # y coordinate of the customer
all_courier_distances = [] # euclidean distance from the courier to the customer

# Standard loop over gym environment 
for i in range(NUM_EPISODES):
    distances = []
    obs = env.reset()
    all_x_coords.append(obs[:,0])
    all_y_coords.append(obs[:,1])
    all_courier_distances.append(get_distance(obs))
    done = False
    timesteps = [0]
    if i > 0:
        all_state_id.append(all_state_id[-1] + 1)
    while not done:
        action = act_greedily(obs)
        # Reward is the negative distance travelled
        obs, reward, done, _ = env.step(action)
        distances.append(-reward)
        if not done:
            timesteps.append(timesteps[-1] + 1)
            all_x_coords.append(obs[:,0])
            all_y_coords.append(obs[:,1])
            all_courier_distances.append(get_distance(obs))
            all_state_id.append(all_state_id[-1] + 1)
    distances_to_go = np.cumsum(distances[::-1])[::-1]
    all_timesteps.extend(timesteps)
    all_distances_to_go.extend(distances_to_go)

states_df = pd.DataFrame({"state_id": all_state_id,
                       "timestep": all_timesteps,
                       "distances_to_go": all_distances_to_go})
objects_df = pd.DataFrame({"state_id": all_state_id,
                        "x_coordinate": all_x_coords,
                        "y_coordinate": all_y_coords,
                        "distance_from_courier": all_courier_distances})

objects_df = objects_df.explode(["x_coordinate", "y_coordinate", "distance_from_courier"], ignore_index=True)

objects_df.index.name = "object_id"

objects_df = objects_df.infer_objects()

# Save the csv(s) to a folder for future use
states_df.to_csv("data/greedy_heuristic_data-states-episodes-" + str(NUM_EPISODES) + "-nodes-" + str(NUM_NODES) + "-seed-" + str(ENV_SEED) + ".csv", index=False)

objects_df.to_csv("data/greedy_heuristic_data-objects-episodes-" + str(NUM_EPISODES) + "-nodes-" + str(NUM_NODES) + "-seed-" + str(ENV_SEED) + ".csv")
