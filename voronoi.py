import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from pathlib import Path
import os

from shapely.geometry import Point, Polygon
from shapely import voronoi_polygons, MultiPoint, normalize

from numpy.typing import NDArray
from typing import Any

X_MIN = -4500
X_MAX = 4500
Y_MIN = -3000
Y_MAX = 3000

PLOT_FREQ = 2000


def plot_voronoi_diagram(points1: NDArray, points2: NDArray, file_name, time_step) -> Any:
    shapely_points = MultiPoint([Point(x, y)
                                for x, y in np.vstack([points1, points2]) if (not math.isinf(x) or not math.isinf(y))])
    boundary = Polygon([(X_MIN, Y_MIN), (X_MIN, Y_MAX),
                       (X_MAX, Y_MAX), (X_MAX, Y_MIN)])

    voronoi_result = voronoi_polygons(
        shapely_points, extend_to=boundary)
    areas = np.zeros(2)
    for i, poly in enumerate(voronoi_result.geoms):
        if np.any([poly.contains(Point(x, y)) for x, y in points1]):
            areas[0] += poly.area
        else:
            areas[1] += poly.area
    if time_step % PLOT_FREQ == 0:
        fig, ax = plt.subplots()
        for i, poly in enumerate(voronoi_result.geoms):
            if np.any([poly.contains(Point(x, y)) for x, y in points1]):
                ax.fill(*poly.exterior.xy, "b", alpha=0.5, edgecolor='black')
            else:
                ax.fill(*poly.exterior.xy, "r", alpha=0.5, edgecolor='black')
        for x, y in points1:
            ax.plot(x, y, 'bx')
        for x, y in points2:
            ax.plot(x, y, 'rx')
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(
            f'Voronoi Diagram, blue: {areas[0]/(areas[0]+areas[1])*100:.1f}%, red: {areas[1]/(areas[0]+areas[1])*100:.1f}%')
        plt.savefig(
            f'figures/{file_name.parent.stem}/{file_name.stem}/voronoi_iter{time_step}.png')
        plt.close()
    return areas


def parse_line(line: any) -> NDArray:
    ball_position = np.array(line[0:2])
    robot_position = np.array(line[2:])

    index = np.where(np.isnan(robot_position))[0][0]
    return ball_position, robot_position[:index], robot_position[index+1:]


def plot_area_percentage(area: NDArray, file_name: str) -> None:
    time = np.arange(0, np.size(area, axis=0)) * 0.02
    horizontal = np.ones(np.size(area, axis=0)) * 50

    avg = np.mean(area[:10000])

    plt.figure()
    plt.plot(time, area, 'b-')
    plt.plot(time, horizontal, 'r--')
    plt.xlabel('Time (s)')
    plt.ylabel('Coverage Percentage (%)')
    plt.title(f'Area of the Voronoi Diagram: average: {avg:.3f}')
    plt.savefig(f'figures/{file_name.parent.stem}/{file_name.stem}/area.png')
    plt.close()


def parse_file(file_path: str) -> None:

    df = pd.read_csv(file_path)
    area = np.array([])
    ball_position = np.array([])
    team1_position = np.array([])
    team2_position = np.array([])

    num_rows = df.shape[0]

    for i, row in df.iterrows():
        ball, team1, team2 = parse_line(row)
        ball_position = np.append(ball_position, ball)
        team1_position = np.append(team1_position, team1)
        team2_position = np.append(team2_position, team2)
        temp = plot_voronoi_diagram(
            team1.reshape(-1, 2), team2.reshape(-1, 2), file_path, i)
        area = np.append(area, temp)

    ball_position = ball_position.reshape(-1, 2)
    team1_position = team1_position.reshape(-1, 10)
    team2_position = team2_position.reshape(-1, 10)
    area = area.reshape(-1, 2)

    area_ratio = area[:, 0]/(area[:, 0]+area[:, 1])*100
    plot_area_percentage(area_ratio, file_path)

    spacing = 2
    indexing = int(3000/spacing)

    def get_distance(position):
        dist_array = np.array([])
        p = position[0::spacing, :]
        for i in range(5):
            dist = np.array(list(map(lambda x, y: 0 if np.any(np.isinf((x))) or np.any(np.isinf(y)) else np.linalg.norm(
                x-y), p[:-1, 2*i:2*i+2], p[1:, 2*i:2*i+2])))/1000
            dist_array = np.append(dist_array, dist)
        return dist_array.reshape(-1, 5)

    dist1 = get_distance(team1_position)[:indexing, :]
    dist2 = get_distance(team2_position)[:indexing, :]
    avg1 = np.mean(np.sum(dist1, axis=0))
    avg2 = np.mean(np.sum(dist2, axis=0))
    plt.figure()
    if file_path.parent.stem == 'penalty_striker':
        pi = [3, 4]
        avgp1 = np.cumsum(dist1[:, pi[0]])
        avgp2 = np.cumsum(dist2[:, pi[1]])
        plt.plot(np.arange(np.shape(dist1)[
                 0])*0.02*spacing, avgp1, 'b.-', label=f'Team1 pstriker: {avgp1[-1]:.2f}')
        plt.plot(np.arange(np.shape(dist1)[
                 0])*0.02*spacing, avgp2, 'r.-', label=f'Team2 pstriker: {avgp2[-1]:.2f}')
    elif file_path.parent.stem == 'penalty_defender':
        pi = [1, 2]
        avgp1 = np.cumsum(dist1[:, pi[0]])
        avgp2 = np.cumsum(dist2[:, pi[1]])
        plt.plot(np.arange(np.shape(dist1)[
                 0])*0.02*spacing, avgp1, 'b.-', label=f'Team1 pstriker: {avgp1[-1]:.2f}')
        plt.plot(np.arange(np.shape(dist1)[
                 0])*0.02*spacing, avgp2, 'r.-', label=f'Team2 pstriker: {avgp2[-1]:.2f}')
    plt.plot(0,
             avg1, 'b-', label=f'Team 1: {avg1:.2f}')
    plt.plot(0,
             avg2, 'r-', label=f'Team 2: {avg2:.2f}')
    plt.xlabel('Time (s)')
    plt.ylabel('Walking Distance (m)')
    plt.title('Average Walking Distance')
    plt.legend()
    plt.savefig(
        f'figures/{file_path.parent.stem}/{file_path.stem}/walking_distance.png')
    plt.close()


if __name__ == '__main__':
    test_cases = ['penalty_striker', 'penalty_defender','coop_gameplay','search','penalty']

    # create figure directories for each test case

    for test_case in test_cases:
        root_folder = f'data/{test_case}/'
        for file in os.listdir(root_folder):
            file_name = file.split('.')[0]
            if not os.path.exists(f'figures/{test_case}/{file_name}/'):
                os.makedirs(f'figures/{test_case}/{file_name}/')
            file_path = Path(f'{root_folder}{file}')
            parse_file(file_path)
