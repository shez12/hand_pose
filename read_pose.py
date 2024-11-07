import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def read_npy_and_separate(file_path):
    data = np.load(file_path)  # Load the .npy file
    separated_data = []  # Initialize a list to hold separated data
    current_chunk = []  # Temporary list to hold current chunk

    for item in data:
        if item.tolist() == [-1, -1, -1]:  # Check for the delimiter
            if current_chunk:  # If current_chunk is not empty
                separated_data.append(current_chunk)  # Append the chunk
                current_chunk = []  # Reset for the next chunk
        else:
            current_chunk.append(item.tolist())  # Add item to the current chunk

    if current_chunk:  # Append any remaining data
        separated_data.append(current_chunk)

    return separated_data  # Return the separated data


def draw_hand_pose(data):
    plt.figure()  # Create a new figure
    ax = plt.axes(projection='3d')  # Create a 3D axis
    data2 =[data[4],data[8],data[12],data[16],data[20]]
    data = [item for item in data if item not in data2]
    # Plot all points in one call
    ax.scatter([point[0] for point in data], 
               [point[1] for point in data], 
               [point[2] for point in data], 
               c='b', marker='o')  

    ax.scatter([point[0] for point in data2], 
               [point[1] for point in data2], 
               [point[2] for point in data2], 
               c='r', marker='x')  

    plt.show()
    plt.close()  # Close the figure after showing




if __name__ == "__main__":
    separated_data = read_npy_and_separate("hand_pose/landmarks1730703567.5929418.npy")
    for data in separated_data:
        draw_hand_pose(data)



