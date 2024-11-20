"""
In this script, we demonstrate how to use DINOBot to do one-shot imitation learning.
You first need to install the following repo and its requirements: https://github.com/ShirAmir/dino-vit-features.
You can then run this file inside that repo.

There are a few setup-dependent functions you need to implement, like getting an RGBD observation from the camera
or moving the robot, that you will find on top of this file.
"""
import cv2
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torchvision import transforms,utils
from PIL import Image
import torchvision.transforms as T
import warnings 
import glob
import time
import rospy
import sys
warnings.filterwarnings("ignore")

from finger_5to2 import *
from dino_vit_features.correspondences import find_correspondences, draw_correspondences
from dino_vit_features.extractor import ViTExtractor
from my_utils.myImageSaver import MyImageSaver
from my_utils.pose_util import *
from my_utils.myRobotSaver import MyRobotSaver,replay_movement,read_movement
from my_utils.depthUtils import project_to_3d
from key_point import extract_keypoints,find_checkpoint
from my_utils.ransac import ransac
from light_glue_points import find_glue_points,select_object


sys.path.append('/home/hanglok/work/ur_slam')
from ik_step import init_robot
import ros_utils.myGripper



#Hyperparameters for DINO correspondences extraction
num_pairs = 10
load_size = 240 #None means no resizing
layer = 9 
facet = 'key' 
bin=True 
thresh=0.05 
model_type='dino_vits8' 
stride=4 
best_joints = [-0.432199780141012, -0.6554244200335901, 1.9931893348693848, -2.4746230284320276, 2.2496132850646973, -2.539410654698507]
intrinsics = [ 912.2659912109375, 911.6720581054688, 637.773193359375, 375.817138671875]

#Deployment hyperparameters    
ERR_THRESHOLD = 0.02 #A generic error between the two sets of points


#Here are the functions you need to create based on your setup.
def camera_get_rgbd(imagesaver):
    '''
    args:
        imagesaver: class of MyImageSaver
    return:
        rgb: rgb image
        depth: depth image
    '''
    rgb_image = imagesaver.rgb_image
    depth_image = imagesaver.depth_image
    if rgb_image is not None and depth_image is not None:
        imagesaver.record()
        return rgb_image, depth_image
    else:
        raise ValueError("No image received from the camera")

    
def robot_move(robot,t_meters,R):
    """
    Inputs: t_meters: (x,y,z) translation in end-effector frame
            R: (3x3) array - rotation matrix in end-effector frame
            robot: class of robot controller
    
    Moves and rotates the robot according to the input translation and rotation.
    """



    transformations = SE3.Ry(90,unit='deg') * SE3.Rz(90,unit='deg')
    pose = Rt_to_pose(R,t_meters)
    SE3_pose = pose_to_SE3(pose)
    SE3_pose = transformations * SE3_pose * transformations.inv()
 
    return robot.step_in_ee(action=SE3_pose,wait=True,return_on_exceed = True)



def record_demo(robot,robot_name,filename):
    recorder = MyRobotSaver(robot,robot_name, filename, init_node=False)
    recorder.record_movement()



def replay_demo(robot,filename):
    '''
    args:
        robot: class of robot controller
        filename: string, path to the file containing the recorded movement
    '''

    positions, velocities,transformations = read_movement(filename)
    replay_movement(robot, positions, velocities,transformations,move_to_start=False)


def replay2(robot,gripper):
    file_path=r'hand_pose/landmarks1730710777.5934556.npy'
    dataset = check_finger_pose(file_path)
    gripper_threshold = 6
    for i in range(len(dataset)-1):
        p1 = np.array(dataset[i][0])
        p2 = np.array(dataset[i][1])
        dist =  np.linalg.norm(p2-p1)*100
        input("break")
        if abs(dist) < gripper_threshold:
            gripper_value  = abs(dist)/6 * 1000
            print("gripper!",gripper_value)
            gripper.set_gripper(gripper_value,5)


        R,t = find_transformation(dataset[i],dataset[i+1])
        print("R\n",R)
        print("t\n",t)
        t =[0,0,0]
        pose = Rt_to_pose(R,t)
        SE3_pose = pose_to_SE3(pose)
        r,p,y = se3_to_rpy(SE3_pose)
        SE3_pose = rpy_to_se3(r,p,y)

        robot.step(action= SE3_pose,wait=True)
    
    gripper.set_gripper(1000,5)




def find_transformation(X, Y):

    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t

def compute_error(points1, points2):
    return np.linalg.norm(np.array(points1) - np.array(points2))

def compute_rgb_error(match_point1, match_point2):
    # Ensure both match_point1 and match_point2 are converted to numpy arrays
    if isinstance(match_point1, torch.Tensor):
        match_point1 = match_point1.cpu().numpy()
    elif isinstance(match_point1, list):
        match_point1 = np.array([p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in match_point1])

    if isinstance(match_point2, torch.Tensor):
        match_point2 = match_point2.cpu().numpy()
    elif isinstance(match_point2, list):
        match_point2 = np.array([p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in match_point2])


    
    # Calculate the error as the max Euclidean distance
    error_list = []
    for i in range(len(match_point1)):
        error_list.append(np.linalg.norm(match_point1[i] - match_point2[i]))
    return max(error_list)

def filter_points(points1, points2):
    '''
    Remove points that are [0,0,0]
    '''
    new_points1 = []
    new_points2 = []
    for i in range(len(points1)):
        if sum(points1[i]) != 0 and sum(points2[i]) != 0:
            new_points1.append(points1[i])
            new_points2.append(points2[i])
    return new_points1, new_points2





def robot_move_version(robot,imagesaver,intrinsics):

    #free cuda cache
    torch.cuda.empty_cache()
    # RECORD DEMO:
    # Move the end-effector to the bottleneck pose and store observation.
    #Get rgbd from wrist camera.
    rgb_bn, depth_bn = camera_get_rgbd(imagesaver)

    print("current pose: ", pose_to_SE3(robot.get_pose()))

    #Record demonstration.
    record_demo(robot,robot_name='robot1',filename='robot1_movements.json')

    input("Press Enter to continue...")
    # TEST TIME DEPLOYMENT
    # Move/change the object and move the end-effector to the home (or a random) pose.
    # while 1:
    error = 100000
    while error > ERR_THRESHOLD:
        #Collect observations at the current pose.
        rgb_live, depth_live = camera_get_rgbd(imagesaver)


        #Compute pixel correspondences between new observation and bottleneck observation.
        with torch.no_grad():
            points1, points2, image1_pil, image2_pil, = find_correspondences(rgb_bn, rgb_live, num_pairs, load_size, layer,
                                                                                facet, bin, thresh, model_type, stride)

            torch.cuda.empty_cache()  # Clear cache after heavy operations
        #Given the pixel coordinates of the correspondences, and their depth values,
        #project the points to 3D space.
        
        
        fig_1, ax1 = plt.subplots()
        ax1.axis('off')
        ax1.imshow(image1_pil)
        fig_2, ax2 = plt.subplots()
        ax2.axis('off')
        ax2.imshow(image2_pil)
        fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
        plt.show()


        points1 = project_to_3d(points1, depth_bn, intrinsics)

        points2 = project_to_3d(points2, depth_live, intrinsics)


        points1, points2 = filter_points(points1, points2)
        print("points1: ", points1)
        print("points2: ", points2)

        # if len(points1) < num_pairs-1:
        #     print("Not enough points")
        #     continue

        #Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation(points1, points2)
        print("moving R: ",R)
        print("moving t: ",t)
  
        error = compute_error(points1, points2)
        print("Error: ", error)

        if error < ERR_THRESHOLD:
            break
        #Move robot

        robot_move(robot,t,R)

        torch.cuda.empty_cache()

        #Once error is small enough, replay demo.
    replay_demo(robot,'records/robot1_movements.json')






def object_move_version(robot,imagesaver,intrinsics):
    '''
    let robot to track moving object 
    '''
    # move robot to the initial pose
    # 初始化窗口
    torch.cuda.empty_cache()
    rgb_bn, depth_bn = camera_get_rgbd(imagesaver)
    # save current joints
    init_joints =  robot.get_joints()

    #Record demonstration.
    record_demo(robot,robot_name='robot1',filename='robot1_movements.json')

    robot.move_joints(init_joints, duration = 0.5, wait=True)

    input("Press Enter to continue...")
    # move to the initial pose


    masks = select_object(rgb_bn)
    
    error = 100000
    while error > ERR_THRESHOLD:
        # move to best obersevation position
        #Collect observations at the current pose.
        rgb_live, depth_live = camera_get_rgbd(imagesaver)
        count = 0
        point_clouds1 = []
        point_clouds2 = []
        match_point_overall_1 = []
        match_point_overall_2 = []
        while count<5:
            #Compute pixel correspondences between new observation and bottleneck observation.
            match_point1,match_point2 = find_glue_points(rgb_bn, rgb_live,masks)
            match_point_overall_1 += match_point1
            match_point_overall_2 += match_point2
            #Given the pixel coordinates of the correspondences, and their depth values,
            #project the points to 3D space
            point_clouds1 += project_to_3d(match_point1, depth_bn, intrinsics,show=False,resize=True ,sequence="xy").copy()
            point_clouds2 += project_to_3d(match_point2,depth_live,intrinsics,show=False,resize=True ,sequence="xy" ).copy()
            count += 1
        origin_points, new_points = filter_points(point_clouds1, point_clouds2)

        # filter outliers
        inliner_mask1 = ransac(origin_points)
        inliner_mask2 = ransac(new_points)
        overall_mask = inliner_mask1 & inliner_mask2
        origin_points = np.array(origin_points)
        new_points = np.array(new_points)
        origin_points = origin_points[overall_mask]
        new_points = new_points[overall_mask]



        #Find rigid translation and rotation that aligns the points by minimising error, using SVD.
        R, t = find_transformation(origin_points,new_points)
        print("moving R: ",R)
        print("moving t: ",t)
   
        error = compute_rgb_error(match_point_overall_1, match_point_overall_2)
        print("Error: ", error)

        if error < 20:
            break
        #Move robot
        will_move = robot_move(robot,t,R)

        # print("origin_points: ", origin_points)
        # print("new_points: ", new_points)
        torch.cuda.empty_cache()
        # input("Press Enter to continue...")
    #Once error is small enough, replay demo.
    replay_demo(robot,'records/robot1_movements.json')
    # ADD GRIPPER CONTROL....
    gripper.set_gripper(700,5)
    input("Press Enter to continue...")
    robot.move_joints(init_joints, duration = 0.5, wait=True)
    replay_demo(robot,'records/robot1_movements.json')
    gripper.set_gripper(1000,5)
    




if __name__ == "__main__":
    #object move
    rospy.init_node('dino_bot')
    gripper = ros_utils.myGripper.MyGripper()
    robot = init_robot("robot1",rotate=False)
    imagesaver = MyImageSaver(cameraNS="camera1")#     imagesaver = MyImageSaver(cameraNS="camera1")

    object_move_version(robot,imagesaver,intrinsics)









# if __name__ == "__main__":
#     rospy.init_node('dino_bot')
#     robot = init_robot("robot1",rotate=False)
#     gripper = myGripper.MyGripper()
#     imagesaver = MyImageSaver(cameraNS="camera1")
#     intrinsics = [ 912.2659912109375, 911.6720581054688, 637.773193359375, 375.817138671875]
#     robot_move_version(robot,imagesaver,intrinsics)
#     robot = init_robot("robot1")
#     replay2(robot,gripper)










# if __name__ == "__main__":
#     rospy.init_node('dino_bot', anonymous=True)
#     imagesaver = MyImageSaver(cameraNS="camera1")
#     rgb_image, depth_image = camera_get_rgbd(imagesaver)
#     print(rgb_image.shape)
#     print(depth_image.shape)
#     intrinsics = [ 912.2659912109375, 911.6720581054688, 637.773193359375, 375.817138671875]
#     points = project_to_3d([[1,1]],depth_image,intrinsics)
#     print(points)























