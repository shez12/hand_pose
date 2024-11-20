import cv2
import time
import numpy as np
import sys
import rospy

from my_utils.pose_util import *
from gesture import detect_all_finger_state, judge_data, get_gripper_coordinates
sys.path.append('/home/hanglok/work/ur_slam')
from ik_step import init_robot
from ros_utils import myGripper

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



def check_finger_pose(file_path):
    '''
    return: 
    dataset: list of 3d coordinates of the tip of the fingers 
    
    '''

    # 使用numpy的load函数读取.npy文件
    data = read_npy_and_separate(file_path)
    # print(data[0])

    h,w=480,640
    
    print(len(data[0]))

    dataset =[]

    for i in range(len(data)):
        data_new=[]
        for j in data[i]:
            data_new.append((int(j[0]*w),int(j[1]*h)))# to 


        # 调用函数，判断每根手指的弯曲或伸直状态
        bend_states, straighten_states = detect_all_finger_state(data_new)



        # 调用函数，检测当前手势
        current_state = judge_data(bend_states, straighten_states,show="False")

        if current_state==True:


            print("current state",current_state)
            data_temp = [data[i][4],data[i][8],data[i][12]]
            print("tips :", data_temp ,"\n")
            dataset.append(data_temp)
        else: 
            print(False)
    return dataset



if __name__ == "__main__":
    file_path=r'hand_pose/landmarks1731660209.389856.npy'
    dataset = check_finger_pose(file_path)
    rospy.init_node('dino_bot')
    robot = init_robot("robot1")
    gripper = myGripper.MyGripper()

    gripper_threshold = 6
    for i in range(len(dataset)-1):
        p1 = np.array(dataset[i][0])
        p2 = np.array(dataset[i][1])
        dist =  np.linalg.norm(p2-p1)*100
        input("break")
        if abs(dist) < gripper_threshold:
            gripper_value  = abs(dist)/6 * 1000
            print("gripper!",gripper_value)
            gripper.set_gripper(1000,5)


        R,t = find_transformation(dataset[i],dataset[i+1])
        print("R\n",R)
        print("t\n",t)
        t =[0,0,0]
        pose = Rt_to_pose(R,t)
        SE3_pose = pose_to_SE3(pose)
        r,p,y = se3_to_rpy(SE3_pose)
        SE3_pose = rpy_to_se3(r,p,y)

        robot.step_ee(action= SE3_pose,wait=True)
    
    gripper.set_gripper(1000,5)