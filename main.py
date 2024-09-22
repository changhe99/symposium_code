import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Ellipse
import matplotlib.patheffects as path_effects

from numpy.typing import NDArray

X_MIN = -4500
X_MAX = 4500
Y_MIN = -3000
Y_MAX = 3000


ALGO = 'new'
SAVEFIG_PATH = f'figures/{ALGO}/'


def get_walking_distance(robot_poses: NDArray) -> float:
    distances = np.array([np.linalg.norm(robot_poses[i, :] - robot_poses[i + 1, :])
                         for i in range(int(np.size(robot_poses, axis=0)-1))])
    total_distance = np.cumsum(distances) / 1000
    return total_distance


def plot_trajectory(ball_trajectory: NDArray, team1_trajectory: NDArray, team2_trajectory: NDArray) -> None:
    plt.figure()
    plt.plot(ball_trajectory[:, 0], ball_trajectory[:, 1], 'r')
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.title('Trajectory of the ball')
    plt.savefig(SAVEFIG_PATH + 'ball_trajectory.png')

    plt.figure()
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    for i in range(5):
        plt.plot(team1_trajectory[:, i, 0], team1_trajectory[:, i, 1], 'b')
        plt.plot(team2_trajectory[:, i, 0], team2_trajectory[:, i, 1], 'r')
    plt.title('Overall trajectory of all robots')
    plt.savefig(SAVEFIG_PATH + 'robot_trajectory.png')


def plot_walking_distance(team_trajectory: NDArray, team: str) -> None:
    time = np.arange(0, np.size(team_trajectory, axis=0)-1) * 0.02

    plt.figure()
    for i in range(5):
        dist = get_walking_distance(team_trajectory[:, i, :])
        plt.plot(time, dist)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Cumulative walking distance of each robot')
    plt.savefig(SAVEFIG_PATH + f'{team}_walking_distance.png')


def plot_distance_comparison(team1_trajectory: NDArray, team2_trajectory: NDArray) -> None:
    time = np.arange(0, np.size(team1_trajectory, axis=0)-1) * 0.02

    dist1 = np.zeros((np.size(team1_trajectory, axis=0)-1))
    dist2 = np.zeros((np.size(team2_trajectory, axis=0)-1))

    plt.figure()
    for i in range(5):
        dist1 += get_walking_distance(team1_trajectory[:, i, :])
        dist2 += get_walking_distance(team2_trajectory[:, i, :])
    plt.plot(time, dist1, 'b')
    plt.plot(time, dist2, 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Cumulative walking distance of each robot')
    plt.savefig(SAVEFIG_PATH + 'walking_distance_comparison.png')


def draw_pitch(dpi=100):
    """Sets up field
    Returns matplotlib fig and axes objects.
    """

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)  # (X_SIZE/10, Y_SIZE/10)
    # complementary: #80a260 e #95bbbc, opposing: #bc95a8 & #bc9f95
    fig.patch.set_facecolor('#a8bc95')

    axes = fig.add_subplot(1, 1, 1)
    axes.set_axis_off()
    axes.set_facecolor('#a8bc95')
    axes.xaxis.set_visible(True)
    axes.yaxis.set_visible(True)

    axes.set_xlim(X_MIN, X_MAX)
    axes.set_ylim(Y_MIN, Y_MAX)

    # plt.xlim([-13.32, 113.32])
    # plt.ylim([-5, 105])

    fig.tight_layout(pad=3)

    draw_patches(axes)

    return fig, axes


def draw_patches(axes):
    """
    Draws basic field shapes on an axes
    """
    # pitch
    axes.add_patch(plt.Rectangle((-4500, -3000), 9000, 6000,
                                 edgecolor="white", facecolor="none"))

    # half-way line
    axes.add_line(plt.Line2D([0, 0], [-3000, 3000],
                             c='w'))

    # penalty areas
    axes.add_patch(plt.Rectangle((),  BOX_WIDTH, BOX_HEIGHT,
                                 ec='w', fc='none'))
    axes.add_patch(plt.Rectangle((0, (100-BOX_HEIGHT)/2),  BOX_WIDTH, BOX_HEIGHT,
                                 ec='w', fc='none'))

    # goal areas
    axes.add_patch(plt.Rectangle((100-GOAL_AREA_WIDTH, (100-GOAL_AREA_HEIGHT)/2),  GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT,
                                 ec='w', fc='none'))
    axes.add_patch(plt.Rectangle((0, (100-GOAL_AREA_HEIGHT)/2),  GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT,
                                 ec='w', fc='none'))

    # goals
    axes.add_patch(plt.Rectangle((100, (100-GOAL)/2),  1, GOAL,
                                 ec='w', fc='none'))
    axes.add_patch(plt.Rectangle((0, (100-GOAL)/2),  -1, GOAL,
                                 ec='w', fc='none'))

    # halfway circle
    axes.add_patch(Ellipse((50, 50), 2*9.15/X_SIZE*100, 2*9.15/Y_SIZE*100,
                           ec='w', fc='none'))

    return axes


def main():
    if not os.path.exists(SAVEFIG_PATH):
        os.makedirs(SAVEFIG_PATH)

    df = pd.read_csv(f"{ALGO}.csv", header=None)
    ball_positions = df.iloc[:, 0:2].to_numpy()
    team1_positions = df.iloc[:, 2:12].to_numpy().reshape(-1, 5, 2)
    team2_positions = df.iloc[:, 12:22].to_numpy().reshape(-1, 5, 2)

    plot_trajectory(ball_positions, team1_positions, team2_positions)
    plot_walking_distance(team1_positions, 'blue')
    plot_walking_distance(team2_positions, 'red')
    plot_distance_comparison(team1_positions, team2_positions)


if __name__ == '__main__':
    # main()
    draw_pitch()
    plt.plot(0, 0, 'ro')
    plt.show()
