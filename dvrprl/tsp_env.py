from typing import Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from dvrprl.common import ObsType
from dvrprl.vrp_graph import VRPGraph


class TSPEnv(gym.Env):
    """
    TSPEnv implements the Traveling Salesmen Problem
    a special variant of the vehicle routing problem.

    State: Shape (batch_size, num_unvisited_nodes, 4) The third
        dimension is structured as follows:
        [x_coord, y_coord, is_depot, visitable]

    Actions:
    Action is a choice of the next unvisited node to visit.

    Reward: -(distance travelled) in the graph between the current node
        and the next visited node dictated by the action.

    Done: True if all nodes have been visited. The distance travelled back
    to the depot is included in the reward but new customers cannot be
    generated during this final return to the depot.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_nodes: int = 20,
        seed: int = 123,
    ) -> None:
        """
        Args:
            num_nodes: Number of nodes in each generated graph. Defaults to 20.
            seed: Seed of the environment. Defaults to 123.
        """

        self.default_rng = np.random.default_rng(seed)

        self.step_count = 0
        self.num_nodes = num_nodes

        self.generate_graph()

    def step(self, action: int) -> Tuple[ObsType, float, bool, dict]:
        """
        Run the environment one timestep. It's the users responsiblity to
        call reset() when the end of the episode has been reached. Accepts
        an action and return a tuple of (observation, reward, done, info)

        Args:
            actions: index of the node to visit in the next timestep.

        Returns:
             Tuple of the observation, reward, done. The reward is for
             the previous action. If done equals True then the episode is over.
        """

        self.step_count += 1

        # Convert the action chosen by the agent which will be from the unvisted
        # to its true node number.
        action_as_features = self.get_state(remove_masked=True)[action]
        true_actions = self.get_state(remove_masked=False)
        action = np.where(np.all(true_actions == action_as_features, axis=1))[0][0]

        # Update the visited nodes and the graph
        self.visited[action] = 1
        traversed_edge = [int(self.current_location[0]), int(action)]
        self.sampler.visit_edge(traversed_edge[0], traversed_edge[1])

        # Get the distance travelled to the next node
        distance_travelled = self.sampler.get_distance(
            traversed_edge[0], traversed_edge[1]
        )

        # Sample new nodes based on the distance travelled
        if traversed_edge[1] == self.depot:
            n_new_nodes = 0
        else:
            n_new_nodes = np.squeeze(self.default_rng.poisson(distance_travelled))

        for _ in range(n_new_nodes):
            new_node_number = self.sampler.graph.number_of_nodes()
            self.sampler.graph.add_node(
                new_node_number,
                coordinates=self.default_rng.random(size=2),
                depot=False,
                node_color="black",
            )
            self.sampler.graph.add_edges_from(
                list(
                    (new_node_number, i)
                    for i in self.sampler.graph.nodes
                    if i != new_node_number
                ),
                visited=False,
            )
            self.visited = np.append(self.visited, 0)

        self.current_location = [action]

        if traversed_edge[1] != self.depot:
            self.generate_mask()

        done = self.is_done()
        return (
            self.get_state(remove_masked=True),
            -distance_travelled,
            done,
            None,
        )

    def is_done(self) -> bool:
        return np.all(self.visited == 1)

    def get_state(self, remove_masked: bool = False) -> np.ndarray:
        """
        Getter for the current environment state

        Args:
            remove_masked: If True, the nodes that are masked
            by the mask aren't included in the state.

        Returns:
            np.ndarray: Shape (num_graph, num_nodes, 4)
            where the third dimension consists of the
            x, y coordinates, if the node is a depot,
            and if it has been visted yet.
        """

        # generate state (depots not yet set)
        state = np.hstack(
            [
                self.sampler.node_positions,
                np.zeros((self.sampler.graph.number_of_nodes(), 1)),
                np.expand_dims(self.generate_mask(), axis=1),
            ]
        )

        # set depots in state to 1
        state[self.depot, 2] = 1

        mask = self.generate_mask()
        trueidx = np.where(mask == 0)

        state = np.hstack(
            [self.sampler.node_positions,
                np.repeat(self.sampler.node_positions[self.current_location], self.sampler.graph.number_of_nodes(), axis=0),
            ]
        )

        if remove_masked:
            return state[trueidx]
        else:
            return state

    def generate_mask(self):
        """
        Generates a mask of where the nodes marked as 1 cannot
        be visited in the next step according to the env dynamic.

        Returns:
            Returns mask for each (un)visitable node
            in each graph. Shape (batch_size, num_nodes)
        """
        # disallow staying on a depot
        self.visited[self.depot] = 1

        # allow staying on a depot if the graph is solved.
        if np.all(self.visited == 1):
            self.visited[self.depot] = 0

        return self.visited

    def reset(self) -> ObsType:
        """
        Resets the environment.

        Returns:
            State of the environment.
        """

        self.step_count = 0
        self.generate_graph()
        return self.get_state(remove_masked=True)

    def generate_graph(self):
        """
        Generates a VRPGraph with num_nodes.
        Resets the visited nodes to 0.
        """
        self.visited = np.zeros(shape=(self.num_nodes,))
        self.sampler = VRPGraph(
            num_nodes=self.num_nodes,
            num_depots=1,
        )

        # set current location to the depot
        self.depot = self.sampler.depots
        self.current_location = self.depot

    def render(self, mode: str = "human"):
        """
        Visualize one step in the env. Since its batched
        this methods renders n random graphs from the batch.
        """
        return self.sampler.draw()

    def enable_video_capturing(self, video_save_path: str):
        self.video_save_path = video_save_path
        if self.video_save_path is not None:
            self.vid = VideoRecorder(self, self.video_save_path)
            self.vid.frames_per_sec = 1
