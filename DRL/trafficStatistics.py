# Pre-data acquisition
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt


def get_speed_range(file_path):
    data = pd.read_csv(file_path, header=None)
    speed_column = -2
    speed_data = data.iloc[:, speed_column]

    min_speed = speed_data.min()
    max_speed = speed_data.max()

    print("Speed range：(", min_speed, ",", max_speed, ")")

def parameterDetermination():

    k_c, v_f, v_c, avg_density = leftLaneAcqSegment()
    ln2 = sp.log(2)
    m = (2 * ln2) / sp.log(v_f / v_c)

    k = avg_density

    return k_c, k, v_c, m

def leftLaneAcqSegment():
    data = pd.read_csv('dataPreprocess/vehicle_data200_1200.csv', header=None)
    data.columns = ['Time', 'Vehicle_ID', 'Distance_from_Start', 'Lane_ID', 'Vehicle_Speed', 'Vehicle_acceleration']

    data = data[data['Lane_ID'] == 2]

    road_length = 1
    segment_length = road_length / 3

    def get_segment(distance):
        if distance < segment_length:
            return 1
        elif distance < 2 * segment_length:
            return 2
        else:
            return 3

    data['segment'] = data['Distance_from_Start'].apply(get_segment)

    density_data = data.groupby(['Time', 'segment'])['Vehicle_ID'].count().reset_index()
    density_data.columns = ['Time', 'segment', 'vehicle_count']
    density_data['density'] = density_data['vehicle_count'] / segment_length

    avg_density = density_data['density'].mean()

    data = pd.merge(data, density_data[['Time', 'segment', 'density']], on=['Time', 'segment'])

    min_density_data = density_data.groupby('segment')['density'].min().reset_index()
    min_density_data.columns = ['segment', 'min_density']

    low_density_data = pd.merge(data, min_density_data, on='segment')
    low_density_data = low_density_data[low_density_data['density'] == low_density_data['min_density']]
    v_f = low_density_data['Vehicle_Speed'].mean()
    print(f"v_f: {v_f}")

    speed_density_data = data.groupby('density')['Vehicle_Speed'].mean().reset_index()
    density = speed_density_data['density']
    speed = speed_density_data['Vehicle_Speed']

    plt.figure(figsize=(10, 6))
    plt.scatter(density, speed, label='Data', alpha=0.6)
    plt.xlabel('Density')
    plt.ylabel('Vehicle_Speed')
    plt.title('Speed-Density Relationship')

    diff_speed = np.diff(speed)
    k_c_index = np.argmin(diff_speed)
    k_c = density.iloc[k_c_index]
    v_c = speed.iloc[k_c_index]

    print(f"k_c: {k_c}")
    print(f"v_c: {v_c}")

    plt.axvline(x=k_c, color='r', linestyle='--', label=f'Critical Density (k_c={k_c:.3f})')
    plt.axhline(y=v_f, color='g', linestyle='--', label=f'Free Flow Speed (v_f={v_f: .3f})')
    plt.axhline(y=v_c, color='b', linestyle='--', label=f'Speed at k_c (v_c={v_c:.3f})')

    plt.legend()
    plt.show()

    return k_c, v_f, v_c, avg_density

def leftLaneAcq():
    data = pd.read_csv('dataPreprocess/vehicle_data200_1200.csv', header=None)
    data.columns = ['Time', 'Vehicle_ID', 'Distance_from_Start', 'Lane_ID', 'Vehicle_Speed', 'Vehicle_acceleration']

    data = data[data['Lane_ID'] == 2]

    road_length = 1

    density_data = data.groupby('Time')['Vehicle_ID'].count().reset_index()
    density_data.columns = ['Time', 'vehicle_count']
    density_data['density'] = density_data['vehicle_count'] / road_length

    data = pd.merge(data, density_data[['Time', 'density']], on='Time')

    avg_density = density_data['density'].mean()

    min_density = density_data['density'].min()
    low_density_data = data[data['density'] == min_density]
    v_f = low_density_data['Vehicle_Speed'].mean()
    print(f"v_f: {v_f}")

    speed_density_data = data.groupby('density')['Vehicle_Speed'].mean().reset_index()
    density = speed_density_data['density']
    speed = speed_density_data['Vehicle_Speed']

    plt.figure(figsize=(10, 6))
    plt.scatter(density, speed, label='Data', alpha=0.6)
    plt.xlabel('Density')
    plt.ylabel('Speed')
    plt.title('Speed-Density Relationship')

    diff_speed = np.diff(speed)
    k_c_index = np.argmin(diff_speed)
    k_c = density.iloc[k_c_index]
    v_c = speed.iloc[k_c_index]

    print(f"k_c: {k_c}")
    print(f"v_c: {v_c}")

    plt.axvline(x=k_c, color='r', linestyle='--', label=f'Critical Density (k_c={k_c:.3f})')
    plt.axhline(y=v_f, color='g', linestyle='--', label=f'Free Flow Speed (v_f={v_f:.3f})')
    plt.axhline(y=v_c, color='b', linestyle='--', label=f'Speed at k_c (v_c={v_c:.3f})')

    plt.legend()
    plt.show()

    return k_c, v_f, v_c, avg_density

def S3(k_c, v_f, k, m):
    v = v_f/(1 + (k/k_c)**m)**(2/m)
    return v

if __name__ == '__main__':
    file_path = 'dataPreprocess/vehicle_data200_1200.csv'
    get_speed_range(file_path)
    # preDataAcq()
    # preDataAcqSegment()

    # leftLaneAcqSegment()

    leftLaneAcq()

    k_c, v_f, k, m = parameterDetermination()

    laneSpeed = S3(k_c, v_f, k, m)
    print("The recommended speed for the leftmost lane is", laneSpeed)