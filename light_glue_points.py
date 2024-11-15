import sys 
import torch
import cv2


sys.path.append("/home/hanglok/work/LightGlue")
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
from sam_utils import select_point,segment,mask_filter



def select_object(image_array1):
    image_array1 = cv2.resize(image_array1, (320,240))
    points = select_point(image_array1,input_is_image = True)
    masks = segment(image_array1, points,input_is_image = True)

    return masks




def find_glue_points(image_array1,image_array2,masks):
    '''
    args:
    image_array1: image array of the first image
    image_array2: image array of the second image

    returns:
    m_kpts0: matched keypoints of the first image
    m_kpts1: matched keypoints of the second image
    
    
    '''
    image_array1 = cv2.resize(image_array1, (320,240))
    image_array2 = cv2.resize(image_array2, (320,240))



    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor

    matcher = LightGlue(features="superpoint").eval().to(device)




    image0 = load_image(image_array1,input_is_image = True)
    image1 = load_image(image_array2,input_is_image = True)

    # print(image1.shape)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})



    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    # print("kpts0",kpts0,kpts0.shape)
    # print("kpts1",kpts1,kpts1.shape)
    # print("match",matches,matches.shape)

    matches = mask_filter(masks,matches,kpts0)
    # print(len(matches))

    # print("matches",matches)

    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


    # print("mkpts0",m_kpts0)
    # print("mkpts1",m_kpts1)

    return m_kpts0,m_kpts1

if __name__ == "__main__":
    image_path1 = "data/images-20241112-172123/camera1_rgb_0.jpg"
    image_path2 = "data/images-20241112-172123/camera1_rgb_1.jpg"
    image_array1 = cv2.imread(image_path1)
    image_array2 = cv2.imread(image_path2)
    find_glue_points(image_array1,image_array2)