import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import os

def visualize_poses(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return

    timestamps = []
    positions = []
    orientations = []

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV columns: unix_time,ovr_timestamp,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w
                try:
                    pos = [float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])]
                    # Quaternion (x, y, z, w)
                    rot = [float(row['rot_x']), float(row['rot_y']), float(row['rot_z']), float(row['rot_w'])]
                    
                    positions.append(pos)
                    orientations.append(rot)
                    timestamps.append(float(row['ovr_timestamp']))
                except ValueError:
                    continue # Skip bad rows
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not positions:
        print("No valid pose data found.")
        return

    positions = np.array(positions)
    orientations = np.array(orientations)

    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    # Note: Unity uses Y-up, but often for 3D plots Z-up is preferred or we just plot as is.
    # Let's plot as is (Unity coordinates)
    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = positions[:, 2]

    ax.plot(xs, ys, zs, label='HMD Path', linewidth=1, alpha=0.8)

    # Mark Start and End
    ax.scatter(xs[0], ys[0], zs[0], c='green', s=50, label='Start')
    ax.scatter(xs[-1], ys[-1], zs[-1], c='red', s=50, label='End')

    # Add orientation quivers (subsampled)
    # Convert quaternion to forward vector
    # Unity coordinate system: Forward is +Z? Actually let's just use the rotation to transform a forward vector.
    # Standard Unity forward is (0, 0, 1).
    # q * v * q_inverse
    
    step = max(1, len(positions) // 50) # Show ~50 arrows max
    
    u_list = []
    v_list = []
    w_list = []
    x_sub = []
    y_sub = []
    z_sub = []

    for i in range(0, len(positions), step):
        # Quaternion rotation of vector (0, 0, 1)
        # Formula for rotating vector v by quaternion q: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
        
        qx, qy, qz, qw = orientations[i]
        
        # Forward vector in local space (Unity usually +Z forward)
        vx, vy, vz = 0, 0, 1
        
        # Apply rotation
        # https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
        
        # q vector part
        q_vec = np.array([qx, qy, qz])
        uv = np.cross(q_vec, np.array([vx, vy, vz]))
        uuv = np.cross(q_vec, uv)
        
        v_prime = np.array([vx, vy, vz]) + ((uv * qw) + uuv) * 2.0
        
        u_list.append(v_prime[0])
        v_list.append(v_prime[1])
        w_list.append(v_prime[2])
        
        x_sub.append(xs[i])
        y_sub.append(ys[i])
        z_sub.append(zs[i])

    # Plot quivers
    ax.quiver(x_sub, y_sub, z_sub, u_list, v_list, w_list, length=0.1, color='orange', alpha=0.5, label='View Dir')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'HMD Trajectory ({len(positions)} points)')
    ax.legend()

    # Set equal aspect ratio for 3D plot (hacky but works for some matplotlib versions)
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize HMD Poses from CSV')
    parser.add_argument('file', nargs='?', default='hmd_poses.csv', help='Path to hmd_poses.csv')
    args = parser.parse_args()

    visualize_poses(args.file)
