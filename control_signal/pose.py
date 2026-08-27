# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import copy
import logging
import math
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch

from .dwpose import util
from .dwpose.wholebody import Wholebody

def compute_face_aspect_ratio(face_kpts):
    """
    face_kpts: numpy array of shape (68, 2), 68 facial landmarks
    returns: float, height / width
    """
    face_kpts = np.array(face_kpts)

    # Jawline (points 0 to 16)
    jaw_left = face_kpts[0]
    jaw_right = face_kpts[16]
    face_width = np.linalg.norm(jaw_right - jaw_left)

    # Approximate face height: from chin (8) to top of face (between eyes)
    chin = face_kpts[8]
    forehead_center = face_kpts[27]  # top of the nose bridge
    face_height = np.linalg.norm(chin - forehead_center)

    # To include upper forehead, you may optionally extend this with a fixed multiplier
    # face_height *= 1.2  # optional

    aspect_ratio = face_height / face_width if face_width != 0 else 0
    return aspect_ratio

def get_logger(name="essmc2"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        std_handler.setFormatter(formatter)
        std_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(std_handler)
    return logger

class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidates, subsets = self.pose_estimation(oriImg)
            nums_candidates = candidates.shape[0]
            poses = []
            for i in range(nums_candidates):
                candidate = candidates[i][np.newaxis, :, :]
                subset = subsets[i][np.newaxis, :]
                nums, keys, locs = candidate.shape
                candidate[..., 0] /= float(W)
                candidate[..., 1] /= float(H)
                body = candidate[:,:18].copy()
                body = body.reshape(nums*18, locs)
                score = subset[:,:18].copy()

                for i in range(len(score)):
                    for j in range(len(score[i])):
                        if score[i][j] > 0.3:
                            score[i][j] = int(18*i+j)
                        else:
                            score[i][j] = -1

                un_visible = subset<0.3
                candidate[un_visible] = -1

                bodyfoot_score = subset[:,:24].copy()
                for i in range(len(bodyfoot_score)):
                    for j in range(len(bodyfoot_score[i])):
                        if bodyfoot_score[i][j] > 0.3:
                            bodyfoot_score[i][j] = int(18*i+j)
                        else:
                            bodyfoot_score[i][j] = -1
                if -1 not in bodyfoot_score[:,18] and -1 not in bodyfoot_score[:,19]:
                    bodyfoot_score[:,18] = np.array([18.])
                else:
                    bodyfoot_score[:,18] = np.array([-1.])
                if -1 not in bodyfoot_score[:,21] and -1 not in bodyfoot_score[:,22]:
                    bodyfoot_score[:,19] = np.array([19.])
                else:
                    bodyfoot_score[:,19] = np.array([-1.])
                bodyfoot_score = bodyfoot_score[:, :20]

                bodyfoot = candidate[:,:24].copy()

                for i in range(nums):
                    if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
                        bodyfoot[i][18] = (bodyfoot[i][18]+bodyfoot[i][19])/2
                    else:
                        bodyfoot[i][18] = np.array([-1., -1.])
                    if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
                        bodyfoot[i][19] = (bodyfoot[i][21]+bodyfoot[i][22])/2
                    else:
                        bodyfoot[i][19] = np.array([-1., -1.])

                bodyfoot = bodyfoot[:,:20,:]
                bodyfoot = bodyfoot.reshape(nums*20, locs)

                foot = candidate[:,18:24]

                faces = candidate[:,24:92]

                hands = candidate[:,92:113]
                hands = np.vstack([hands, candidate[:,113:]])

                # bodies = dict(candidate=body, subset=score)
                bodies = dict(candidate=bodyfoot, subset=bodyfoot_score)
                pose = dict(bodies=bodies, hands=hands, faces=faces)
                poses.append(pose)

            # return draw_pose(pose, H, W)
            return poses

def draw_poses(poses, H, W, drop_head=False, drop_face=False):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        bodies = pose['bodies']
        faces = pose['faces']
        hands = pose['hands']
        candidate = bodies['candidate']
        subset = bodies['subset']

        canvas = util.draw_body_and_foot(canvas, candidate, subset, drop_head)

        canvas = util.draw_handpose(canvas, hands)

        if not drop_head and not drop_face:
            canvas = util.draw_facepose(canvas, faces)

    return canvas


def align_poses(pose_ref, results_vis, first_frame=False, previous_params=None, use_align=True):

    # import ipdb; ipdb.set_trace()

    bodies = results_vis[0]['bodies']
    faces = results_vis[0]['faces']
    hands = results_vis[0]['hands']
    candidate = bodies['candidate']

    ref_bodies = pose_ref['bodies']
    ref_faces = pose_ref['faces']
    ref_hands = pose_ref['hands']
    ref_candidate = ref_bodies['candidate']


    ref_2_x = ref_candidate[2][0]
    ref_2_y = ref_candidate[2][1]
    ref_5_x = ref_candidate[5][0]
    ref_5_y = ref_candidate[5][1]
    ref_8_x = ref_candidate[8][0]
    ref_8_y = ref_candidate[8][1]
    ref_11_x = ref_candidate[11][0]
    ref_11_y = ref_candidate[11][1]
    ref_center1 = 0.5*(ref_candidate[2]+ref_candidate[5])
    ref_center2 = 0.5*(ref_candidate[8]+ref_candidate[11])

    zero_2_x = candidate[2][0]
    zero_2_y = candidate[2][1]
    zero_5_x = candidate[5][0]
    zero_5_y = candidate[5][1]
    zero_8_x = candidate[8][0]
    zero_8_y = candidate[8][1]
    zero_11_x = candidate[11][0]
    zero_11_y = candidate[11][1]
    zero_center1 = 0.5*(candidate[2]+candidate[5])
    zero_center2 = 0.5*(candidate[8]+candidate[11])

    # first check if all > 0
    x_val_valid = False
    if first_frame:
        if (ref_5_x > 0).all() and (ref_2_x > 0).all() and (zero_5_x > 0).all() and (zero_2_x > 0).all():
            x_ratio = (ref_5_x-ref_2_x)/(zero_5_x-zero_2_x)
            x_val_valid = True
        else:
            x_ratio = 1.0
    elif previous_params is not None:
        x_ratio = previous_params['x_ratio']
        x_val_valid = True
    else:
        x_ratio = 1.0

    y_val_valid = False
    if first_frame:
        y_valid = (ref_candidate[2] > 0).all() and (ref_candidate[5] > 0).all() and (candidate[2] > 0).all() and (candidate[5] > 0).all() \
            and (ref_candidate[8] > 0).all() and (ref_candidate[11] > 0).all() and (candidate[8] > 0).all() and (candidate[11] > 0).all()
        if y_valid:
            y_ratio = (ref_center2[1]-ref_center1[1])/(zero_center2[1]-zero_center1[1])
            y_val_valid = True
        else:
            y_ratio = 1.0
    elif previous_params is not None:
        y_ratio = previous_params['y_ratio']
        y_val_valid = True
    else:
        y_ratio = 1.0

    if x_val_valid and not y_val_valid:
        y_ratio = x_ratio
    elif not x_val_valid and y_val_valid:
        x_ratio = y_ratio

    # use avg of x and y ratio
    x_ratio = (x_ratio + y_ratio) / 2
    y_ratio = x_ratio


    if x_ratio < 1:
        x_ratio = 1
    if y_ratio < 1:
        y_ratio = 1

    # import ipdb; ipdb.set_trace()


    origin_candidate = results_vis[0]['bodies']['candidate'].copy()

    results_vis[0]['bodies']['candidate'][:,0] *= x_ratio
    results_vis[0]['bodies']['candidate'][:,1] *= y_ratio


    results_vis[0]['hands'][:,:,0] *= x_ratio
    results_vis[0]['hands'][:,:,1] *= y_ratio


    ########neck########
    l_neck_ref = ((ref_candidate[0][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[0][1] - ref_candidate[1][1]) ** 2) ** 0.5
    l_neck_0 = ((candidate[0][0] - candidate[1][0]) ** 2 + (candidate[0][1] - candidate[1][1]) ** 2) ** 0.5



    if use_align:
        if first_frame:
            if l_neck_0 == 0 or l_neck_ref == 0:
                neck_ratio = 1
            else:
                neck_ratio = l_neck_ref / l_neck_0
        elif previous_params is not None:
            neck_ratio = previous_params['neck_ratio']
        else:
            neck_ratio = 1
    else:
        neck_ratio = 1

    x_offset_neck = (candidate[1][0]-candidate[0][0])*(1.-neck_ratio)
    y_offset_neck = (candidate[1][1]-candidate[0][1])*(1.-neck_ratio)

    results_vis[0]['bodies']['candidate'][0,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][0,1] += y_offset_neck
    results_vis[0]['bodies']['candidate'][14,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][14,1] += y_offset_neck
    results_vis[0]['bodies']['candidate'][15,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][15,1] += y_offset_neck
    results_vis[0]['bodies']['candidate'][16,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][16,1] += y_offset_neck
    results_vis[0]['bodies']['candidate'][17,0] += x_offset_neck
    results_vis[0]['bodies']['candidate'][17,1] += y_offset_neck

    ########shoulder2########
    l_shoulder2_ref = ((ref_candidate[2][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[2][1] - ref_candidate[1][1]) ** 2) ** 0.5
    l_shoulder2_0 = ((candidate[2][0] - candidate[1][0]) ** 2 + (candidate[2][1] - candidate[1][1]) ** 2) ** 0.5

    if first_frame:
        if l_shoulder2_0 == 0 or l_shoulder2_ref == 0:
            shoulder2_ratio = None
        else:
            shoulder2_ratio = l_shoulder2_ref / l_shoulder2_0
    elif previous_params is not None:
        shoulder2_ratio = previous_params['shoulder2_ratio']
    else:
        shoulder2_ratio = None

    ########shoulder5########
    l_shoulder5_ref = ((ref_candidate[5][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[5][1] - ref_candidate[1][1]) ** 2) ** 0.5
    l_shoulder5_0 = ((candidate[5][0] - candidate[1][0]) ** 2 + (candidate[5][1] - candidate[1][1]) ** 2) ** 0.5

    if first_frame:
        if l_shoulder5_0 == 0 or l_shoulder5_ref == 0:
            shoulder5_ratio = None
        else:
            shoulder5_ratio = l_shoulder5_ref / l_shoulder5_0
    elif previous_params is not None:
        shoulder5_ratio = previous_params['shoulder5_ratio']
    else:
        shoulder5_ratio = None

    if use_align:
        if shoulder2_ratio is None and shoulder5_ratio is None:
            shoulder_ratio = 1
        elif shoulder2_ratio is None:
            shoulder_ratio = shoulder5_ratio
        elif shoulder5_ratio is None:
            shoulder_ratio = shoulder2_ratio
        else:
            shoulder_ratio = (shoulder2_ratio + shoulder5_ratio) / 2
    else:
        shoulder_ratio = 1


    shoulder5_ratio = shoulder_ratio
    shoulder2_ratio = shoulder_ratio

    x_offset_shoulder2 = (candidate[1][0]-candidate[2][0])*(1.-shoulder_ratio)
    y_offset_shoulder2 = (candidate[1][1]-candidate[2][1])*(1.-shoulder_ratio)

    results_vis[0]['bodies']['candidate'][2,0] += x_offset_shoulder2
    results_vis[0]['bodies']['candidate'][2,1] += y_offset_shoulder2
    results_vis[0]['bodies']['candidate'][3,0] += x_offset_shoulder2
    results_vis[0]['bodies']['candidate'][3,1] += y_offset_shoulder2
    results_vis[0]['bodies']['candidate'][4,0] += x_offset_shoulder2
    results_vis[0]['bodies']['candidate'][4,1] += y_offset_shoulder2
    results_vis[0]['hands'][1,:,0] += x_offset_shoulder2
    results_vis[0]['hands'][1,:,1] += y_offset_shoulder2



    x_offset_shoulder5 = (candidate[1][0]-candidate[5][0])*(1.-shoulder_ratio)
    y_offset_shoulder5 = (candidate[1][1]-candidate[5][1])*(1.-shoulder_ratio)

    results_vis[0]['bodies']['candidate'][5,0] += x_offset_shoulder5
    results_vis[0]['bodies']['candidate'][5,1] += y_offset_shoulder5
    results_vis[0]['bodies']['candidate'][6,0] += x_offset_shoulder5
    results_vis[0]['bodies']['candidate'][6,1] += y_offset_shoulder5
    results_vis[0]['bodies']['candidate'][7,0] += x_offset_shoulder5
    results_vis[0]['bodies']['candidate'][7,1] += y_offset_shoulder5
    results_vis[0]['hands'][0,:,0] += x_offset_shoulder5
    results_vis[0]['hands'][0,:,1] += y_offset_shoulder5

    ########arm3######## right upper arm
    l_arm3_ref = ((ref_candidate[3][0] - ref_candidate[2][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[2][1]) ** 2) ** 0.5
    l_arm3_0 = ((candidate[3][0] - candidate[2][0]) ** 2 + (candidate[3][1] - candidate[2][1]) ** 2) ** 0.5

    if first_frame:
        if l_arm3_0 == 0 or l_arm3_ref == 0:
            arm3_ratio = None
        else:
            arm3_ratio = l_arm3_ref / l_arm3_0
    elif previous_params is not None:
        arm3_ratio = previous_params['arm3_ratio']
    else:
        arm3_ratio = None

    ########arm6######## left upper arm
    l_arm6_ref = ((ref_candidate[6][0] - ref_candidate[5][0]) ** 2 + (ref_candidate[6][1] - ref_candidate[5][1]) ** 2) ** 0.5
    l_arm6_0 = ((candidate[6][0] - candidate[5][0]) ** 2 + (candidate[6][1] - candidate[5][1]) ** 2) ** 0.5

    if first_frame:
        if l_arm6_0 == 0 or l_arm6_ref == 0:
            arm6_ratio = None
        else:
            arm6_ratio = l_arm6_ref / l_arm6_0
    elif previous_params is not None:
        arm6_ratio = previous_params['arm6_ratio']
    else:
        arm6_ratio = None

    if use_align:
        if arm3_ratio is None and arm6_ratio is None:
            upper_arm_ratio = 1
        elif arm3_ratio is None:
            upper_arm_ratio = arm6_ratio
        elif arm6_ratio is None:
            upper_arm_ratio = arm3_ratio
        else:
            upper_arm_ratio = (arm3_ratio + arm6_ratio) / 2
    else:
        upper_arm_ratio = 1

    arm3_ratio = upper_arm_ratio
    arm6_ratio = upper_arm_ratio


    x_offset_arm3 = (candidate[2][0]-candidate[3][0])*(1.-upper_arm_ratio)
    y_offset_arm3 = (candidate[2][1]-candidate[3][1])*(1.-upper_arm_ratio)

    results_vis[0]['bodies']['candidate'][3,0] += x_offset_arm3
    results_vis[0]['bodies']['candidate'][3,1] += y_offset_arm3
    results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm3
    results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm3
    results_vis[0]['hands'][1,:,0] += x_offset_arm3
    results_vis[0]['hands'][1,:,1] += y_offset_arm3



    x_offset_arm6 = (candidate[5][0]-candidate[6][0])*(1.-upper_arm_ratio)
    y_offset_arm6 = (candidate[5][1]-candidate[6][1])*(1.-upper_arm_ratio)

    results_vis[0]['bodies']['candidate'][6,0] += x_offset_arm6
    results_vis[0]['bodies']['candidate'][6,1] += y_offset_arm6
    results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm6
    results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm6
    results_vis[0]['hands'][0,:,0] += x_offset_arm6
    results_vis[0]['hands'][0,:,1] += y_offset_arm6

    ########arm4######## right front arm
    l_arm4_ref = ((ref_candidate[4][0] - ref_candidate[3][0]) ** 2 + (ref_candidate[4][1] - ref_candidate[3][1]) ** 2) ** 0.5
    l_arm4_0 = ((candidate[4][0] - candidate[3][0]) ** 2 + (candidate[4][1] - candidate[3][1]) ** 2) ** 0.5

    if first_frame:
        if l_arm4_0 == 0 or l_arm4_ref == 0:
            arm4_ratio = None
        else:
            arm4_ratio = l_arm4_ref / l_arm4_0
    elif previous_params is not None:
        arm4_ratio = previous_params['arm4_ratio']
    else:
        arm4_ratio = None
    ########arm7######## left front arm
    l_arm7_ref = ((ref_candidate[7][0] - ref_candidate[6][0]) ** 2 + (ref_candidate[7][1] - ref_candidate[6][1]) ** 2) ** 0.5
    l_arm7_0 = ((candidate[7][0] - candidate[6][0]) ** 2 + (candidate[7][1] - candidate[6][1]) ** 2) ** 0.5

    if first_frame:
        if l_arm7_0 == 0 or l_arm7_ref == 0:
            arm7_ratio = None
        else:
            arm7_ratio = l_arm7_ref / l_arm7_0
    elif previous_params is not None:
        arm7_ratio = previous_params['arm7_ratio']
    else:
        arm7_ratio = None

    if use_align:
        if arm4_ratio is None and arm7_ratio is None:
            front_arm_ratio = 1
        elif arm4_ratio is None:
            front_arm_ratio = arm7_ratio
        elif arm7_ratio is None:
            front_arm_ratio = arm4_ratio
        else:
            front_arm_ratio = (arm4_ratio + arm7_ratio) / 2
    else:
        front_arm_ratio = 1

    arm4_ratio = front_arm_ratio
    arm7_ratio = front_arm_ratio

    x_offset_arm4 = (candidate[3][0]-candidate[4][0])*(1.-front_arm_ratio)
    y_offset_arm4 = (candidate[3][1]-candidate[4][1])*(1.-front_arm_ratio)

    results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm4
    results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm4
    results_vis[0]['hands'][1,:,0] += x_offset_arm4
    results_vis[0]['hands'][1,:,1] += y_offset_arm4





    x_offset_arm7 = (candidate[6][0]-candidate[7][0])*(1.-front_arm_ratio)
    y_offset_arm7 = (candidate[6][1]-candidate[7][1])*(1.-front_arm_ratio)

    results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm7
    results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm7
    results_vis[0]['hands'][0,:,0] += x_offset_arm7
    results_vis[0]['hands'][0,:,1] += y_offset_arm7

    ########head14######## right eye
    l_head14_ref = ((ref_candidate[14][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[14][1] - ref_candidate[0][1]) ** 2) ** 0.5
    l_head14_0 = ((candidate[14][0] - candidate[0][0]) ** 2 + (candidate[14][1] - candidate[0][1]) ** 2) ** 0.5

    if first_frame:
        if l_head14_0 == 0 or l_head14_ref == 0:
            head14_ratio = None
        else:
            head14_ratio = l_head14_ref / l_head14_0
    elif previous_params is not None:
        head14_ratio = previous_params['head14_ratio']
    else:
        head14_ratio = None

    ########head15######## left eye
    l_head15_ref = ((ref_candidate[15][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[15][1] - ref_candidate[0][1]) ** 2) ** 0.5
    l_head15_0 = ((candidate[15][0] - candidate[0][0]) ** 2 + (candidate[15][1] - candidate[0][1]) ** 2) ** 0.5

    if first_frame:
        if l_head15_0 == 0 or l_head15_ref == 0:
            head15_ratio = None
        else:
            head15_ratio = l_head15_ref / l_head15_0
    elif previous_params is not None:
        head15_ratio = previous_params['head15_ratio']
    else:
        head15_ratio = None


    if use_align:
        if head14_ratio is None and head15_ratio is None:
            eye_ratio = 1
        elif head14_ratio is None:
            eye_ratio = head15_ratio
        elif head15_ratio is None:
            eye_ratio = head14_ratio
        else:
            eye_ratio = (head14_ratio + head15_ratio) / 2
    else:
        eye_ratio = 1

    head14_ratio = eye_ratio
    head15_ratio = eye_ratio

    x_offset_head14 = (candidate[0][0]-candidate[14][0])*(1.-eye_ratio)
    y_offset_head14 = (candidate[0][1]-candidate[14][1])*(1.-eye_ratio)

    results_vis[0]['bodies']['candidate'][14,0] += x_offset_head14
    results_vis[0]['bodies']['candidate'][14,1] += y_offset_head14
    results_vis[0]['bodies']['candidate'][16,0] += x_offset_head14
    results_vis[0]['bodies']['candidate'][16,1] += y_offset_head14



    x_offset_head15 = (candidate[0][0]-candidate[15][0])*(1.-eye_ratio)
    y_offset_head15 = (candidate[0][1]-candidate[15][1])*(1.-eye_ratio)

    results_vis[0]['bodies']['candidate'][15,0] += x_offset_head15
    results_vis[0]['bodies']['candidate'][15,1] += y_offset_head15
    results_vis[0]['bodies']['candidate'][17,0] += x_offset_head15
    results_vis[0]['bodies']['candidate'][17,1] += y_offset_head15

    ########head16######## right ear
    l_head16_ref = ((ref_candidate[16][0] - ref_candidate[14][0]) ** 2 + (ref_candidate[16][1] - ref_candidate[14][1]) ** 2) ** 0.5
    l_head16_0 = ((candidate[16][0] - candidate[14][0]) ** 2 + (candidate[16][1] - candidate[14][1]) ** 2) ** 0.5

    if first_frame:
        if l_head16_0 == 0 or l_head16_ref == 0:
            head16_ratio = None
        else:
            head16_ratio = l_head16_ref / l_head16_0
    elif previous_params is not None:
        head16_ratio = previous_params['head16_ratio']
    else:
        head16_ratio = None



    ########head17######## left ear
    l_head17_ref = ((ref_candidate[17][0] - ref_candidate[15][0]) ** 2 + (ref_candidate[17][1] - ref_candidate[15][1]) ** 2) ** 0.5
    l_head17_0 = ((candidate[17][0] - candidate[15][0]) ** 2 + (candidate[17][1] - candidate[15][1]) ** 2) ** 0.5

    if first_frame:
        if l_head17_0 == 0 or l_head17_ref == 0:
            head17_ratio = None
        else:
            head17_ratio = l_head17_ref / l_head17_0
    elif previous_params is not None:
        head17_ratio = previous_params['head17_ratio']
    else:
        head17_ratio = None

    if use_align:
        if head16_ratio is None and head17_ratio is None:
            ear_ratio = 1
        elif head16_ratio is None:
            ear_ratio = head17_ratio
        elif head17_ratio is None:
            ear_ratio = head16_ratio
        else:
            ear_ratio = (head16_ratio + head17_ratio) / 2
    else:
        ear_ratio = 1

    head16_ratio = ear_ratio
    head17_ratio = ear_ratio

    x_offset_head16 = (candidate[14][0]-candidate[16][0])*(1.-ear_ratio)
    y_offset_head16 = (candidate[14][1]-candidate[16][1])*(1.-ear_ratio)

    results_vis[0]['bodies']['candidate'][16,0] += x_offset_head16
    results_vis[0]['bodies']['candidate'][16,1] += y_offset_head16



    x_offset_head17 = (candidate[15][0]-candidate[17][0])*(1.-ear_ratio)
    y_offset_head17 = (candidate[15][1]-candidate[17][1])*(1.-ear_ratio)

    results_vis[0]['bodies']['candidate'][17,0] += x_offset_head17
    results_vis[0]['bodies']['candidate'][17,1] += y_offset_head17





    ########MovingAverage########

    ########left leg########
    l_ll1_ref = ((ref_candidate[8][0] - ref_candidate[9][0]) ** 2 + (ref_candidate[8][1] - ref_candidate[9][1]) ** 2) ** 0.5
    l_ll1_0 = ((candidate[8][0] - candidate[9][0]) ** 2 + (candidate[8][1] - candidate[9][1]) ** 2) ** 0.5

    if first_frame:
        if l_ll1_0 == 0 or l_ll1_ref == 0:
            ll1_ratio = None
        else:
            ll1_ratio = l_ll1_ref / l_ll1_0
    elif previous_params is not None:
        ll1_ratio = previous_params['ll1_ratio']
    else:
        ll1_ratio = None



    ########right leg########
    l_rl1_ref = ((ref_candidate[11][0] - ref_candidate[12][0]) ** 2 + (ref_candidate[11][1] - ref_candidate[12][1]) ** 2) ** 0.5
    l_rl1_0 = ((candidate[11][0] - candidate[12][0]) ** 2 + (candidate[11][1] - candidate[12][1]) ** 2) ** 0.5
    if first_frame:
        if l_rl1_0 == 0 or l_rl1_ref == 0:
            rl1_ratio = None
        else:
            rl1_ratio = l_rl1_ref / l_rl1_0
    elif previous_params is not None:
        rl1_ratio = previous_params['rl1_ratio']
    else:
        rl1_ratio = None


    if use_align:
        if ll1_ratio is None and rl1_ratio is None:
            leg_ratio = 1
        elif ll1_ratio is None:
            leg_ratio = rl1_ratio
        elif rl1_ratio is None:
            leg_ratio = ll1_ratio
        else:
            leg_ratio = (ll1_ratio + rl1_ratio) / 2
    else:
        leg_ratio = 1

    ll1_ratio = leg_ratio
    rl1_ratio = leg_ratio

    x_offset_ll1 = (candidate[9][0]-candidate[8][0])*(leg_ratio-1.)
    y_offset_ll1 = (candidate[9][1]-candidate[8][1])*(leg_ratio-1.)

    results_vis[0]['bodies']['candidate'][9,0] += x_offset_ll1
    results_vis[0]['bodies']['candidate'][9,1] += y_offset_ll1
    results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll1
    results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll1
    results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll1
    results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll1

    ########left leg######## sec part


    l_ll2_ref = ((ref_candidate[9][0] - ref_candidate[10][0]) ** 2 + (ref_candidate[9][1] - ref_candidate[10][1]) ** 2) ** 0.5
    l_ll2_0 = ((candidate[9][0] - candidate[10][0]) ** 2 + (candidate[9][1] - candidate[10][1]) ** 2) ** 0.5

    if first_frame:
        if l_ll2_0 == 0 or l_ll2_ref == 0:
            ll2_ratio = None
        else:
            ll2_ratio = l_ll2_ref / l_ll2_0
    elif previous_params is not None:
        ll2_ratio = previous_params['ll2_ratio']
    else:
        ll2_ratio = None

    l_rl2_ref = ((ref_candidate[12][0] - ref_candidate[13][0]) ** 2 + (ref_candidate[12][1] - ref_candidate[13][1]) ** 2) ** 0.5
    l_rl2_0 = ((candidate[12][0] - candidate[13][0]) ** 2 + (candidate[12][1] - candidate[13][1]) ** 2) ** 0.5

    if first_frame:
        if l_rl2_0 == 0 or l_rl2_ref == 0:
            rl2_ratio = None
        else:
            rl2_ratio = l_rl2_ref / l_rl2_0
    elif previous_params is not None:
        rl2_ratio = previous_params['rl2_ratio']
    else:
        rl2_ratio = None

    if use_align:
        if ll2_ratio is None and rl2_ratio is None:
            leg_ratio2 = 1
        elif ll2_ratio is None:
            leg_ratio2 = rl2_ratio
        elif rl2_ratio is None:
            leg_ratio2 = ll2_ratio
        else:
            leg_ratio2 = (ll2_ratio + rl2_ratio) / 2
    else:
        leg_ratio2 = 1

    ll2_ratio = leg_ratio2
    rl2_ratio = leg_ratio2

    x_offset_ll2 = (candidate[10][0]-candidate[9][0])*(leg_ratio2-1.)
    y_offset_ll2 = (candidate[10][1]-candidate[9][1])*(leg_ratio2-1.)

    results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll2
    results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll2
    results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll2
    results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll2



    x_offset_rl1 = (candidate[12][0]-candidate[11][0])*(leg_ratio2-1.)
    y_offset_rl1 = (candidate[12][1]-candidate[11][1])*(leg_ratio2-1.)

    results_vis[0]['bodies']['candidate'][12,0] += x_offset_rl1
    results_vis[0]['bodies']['candidate'][12,1] += y_offset_rl1
    results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl1
    results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl1
    results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl1
    results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl1




    x_offset_rl2 = (candidate[13][0]-candidate[12][0])*(rl2_ratio-1.)
    y_offset_rl2 = (candidate[13][1]-candidate[12][1])*(rl2_ratio-1.)

    results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl2
    results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl2
    results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl2
    results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl2

    if first_frame:
        offset_nose = ref_candidate[0] - results_vis[0]['bodies']['candidate'][0] # use nose instead of neck
        offset_neck = ref_candidate[1] - results_vis[0]['bodies']['candidate'][1]
        # use nose for y, neck for x
        offset = np.array([offset_neck[0], offset_nose[1]])
    elif previous_params is not None:
        offset = previous_params['offset']
    else:
        offset = np.zeros(2)

    results_vis[0]['bodies']['candidate'] += offset[np.newaxis, :]

    # Remove body offset for faces - faces use head-based offset only
    # results_vis[0]['faces'] += offset[np.newaxis, np.newaxis, :]
    results_vis[0]['hands'] += offset[np.newaxis, np.newaxis, :]



    ########Calculate face transformation using head keypoints########
    # Use nose (0), left eye (15), right eye (14), left ear (17), right ear (16)
    # Extract head keypoints from reference and current (after body adjustments)
    head_indices = [0, 14, 15, 16, 17, 1]  # nose, right_eye, left_eye, right_ear, left_ear, neck

    # Calculate face transformation using head keypoints
    origin_head_keypoints = origin_candidate[head_indices]
    target_head_keypoints = results_vis[0]['bodies']['candidate'][head_indices]

    # Filter out invalid keypoints (coordinates < 0)
    valid_orig = origin_head_keypoints[:, 0] >= 0
    valid_target = target_head_keypoints[:, 0] >= 0
    valid_mask = valid_orig & valid_target



    if first_frame:
        if np.sum(valid_mask) < 2:  # Need at least 2 valid points
            raise ValueError("Not enough valid keypoints to calculate face transformation")
        else:
            valid_orig_points = origin_head_keypoints[valid_mask]
            valid_target_points = target_head_keypoints[valid_mask]

            # Calculate centers
            orig_center = np.mean(valid_orig_points, axis=0)
            target_center = np.mean(valid_target_points, axis=0)

            # 簡化方法：用任意兩個有效的頭部關鍵點計算縮放
            # 順序: 左眼, 右眼
            point_orders = [[2, 1], [3, 4]]  # left_eye(15), right_eye(14)

            # 找到第一對有效的點
            orig_pt1, orig_pt2 = None, None
            target_pt1, target_pt2 = None, None
            for point_order in point_orders:
                for i in range(len(point_order)):
                    for j in range(i+1, len(point_order)):
                        idx1, idx2 = point_order[i], point_order[j]
                        if (idx1 < len(valid_mask) and idx2 < len(valid_mask) and
                            valid_mask[idx1] and valid_mask[idx2]):
                            orig_pt1 = origin_head_keypoints[idx1]
                            orig_pt2 = origin_head_keypoints[idx2]
                            target_pt1 = target_head_keypoints[idx1]
                            target_pt2 = target_head_keypoints[idx2]
                            break
                    if orig_pt1 is not None:
                        break

                if orig_pt1 is not None:
                    break

            # import ipdb; ipdb.set_trace()

            if orig_pt1 is not None and orig_pt2 is not None:
                # 用找到的兩個點定義axis
                orig_axis_vector = orig_pt2 - orig_pt1
                target_axis_vector = target_pt2 - target_pt1
                orig_axis_length = np.linalg.norm(orig_axis_vector)
                target_axis_length = np.linalg.norm(target_axis_vector)

                if orig_axis_length > 0 and target_axis_length > 0:
                    # 定義axis和orthogonal axis
                    orig_axis_unit = orig_axis_vector / orig_axis_length
                    target_axis_unit = target_axis_vector / target_axis_length
                    orig_ortho_unit = np.array([-orig_axis_unit[1], orig_axis_unit[0]])
                    target_ortho_unit = np.array([-target_axis_unit[1], target_axis_unit[0]])

                    # 投影所有頭部關鍵點到axis和orthogonal axis
                    orig_center = (orig_pt1 + orig_pt2) / 2
                    target_center = (target_pt1 + target_pt2) / 2

                    orig_axis_coords = []
                    orig_ortho_coords = []
                    target_axis_coords = []
                    target_ortho_coords = []

                    for i in range(len(origin_head_keypoints)):
                        if valid_mask[i]:
                            # 相對於中心的座標
                            orig_rel = origin_head_keypoints[i] - orig_center
                            target_rel = target_head_keypoints[i] - target_center

                            # 投影到axis和orthogonal axis
                            orig_axis_coord = np.dot(orig_rel, orig_axis_unit)
                            orig_ortho_coord = np.dot(orig_rel, orig_ortho_unit)
                            target_axis_coord = np.dot(target_rel, target_axis_unit)
                            target_ortho_coord = np.dot(target_rel, target_ortho_unit)

                            orig_axis_coords.append(orig_axis_coord)
                            orig_ortho_coords.append(orig_ortho_coord)
                            target_axis_coords.append(target_axis_coord)
                            target_ortho_coords.append(target_ortho_coord)


                    orig_axis_range = max(orig_axis_coords) - min(orig_axis_coords)
                    orig_ortho_range = max(orig_ortho_coords) - min(orig_ortho_coords)
                    target_axis_range = max(target_axis_coords) - min(target_axis_coords)
                    # target_ortho_range = max(target_ortho_coords) - min(target_ortho_coords)

                    axis_ratio = target_axis_range / orig_axis_range if orig_axis_range > 0 else 1.0
                    ortho_ratio = axis_ratio
                    # get ref face aspect ratio
                    ref_face_aspect_ratio = compute_face_aspect_ratio(ref_faces[0])
                    # get orign face aspect ratio
                    target_face_aspect_ratio = compute_face_aspect_ratio(results_vis[0]['faces'][0])
                    # get ratio
                    face_ratio = ref_face_aspect_ratio / target_face_aspect_ratio

                    if use_align:
                        ortho_ratio = face_ratio * axis_ratio
                    else:
                        ortho_ratio = axis_ratio

                else:
                    raise ValueError("Not enough valid keypoints to calculate face transformation")


    else:
        axis_ratio = previous_params['axis_ratio']
        ortho_ratio = previous_params['ortho_ratio']
        target_center = previous_params['target_center']
        orig_center = previous_params['orig_center']
        orig_axis_unit = previous_params['orig_axis_unit']
        orig_ortho_unit = previous_params['orig_ortho_unit']
        target_axis_unit = previous_params['target_axis_unit']
        target_ortho_unit = previous_params['target_ortho_unit']

    if len(results_vis[0]['faces']) > 0:
        # import ipdb; ipdb.set_trace()
        # 以原始軸中心為基準進行縮放
        centered_faces = results_vis[0]['faces'] - orig_center

        # 對每個face keypoint進行axis方向縮放
        for i in range(results_vis[0]['faces'].shape[0]):
            for j in range(results_vis[0]['faces'].shape[1]):
                if results_vis[0]['faces'][i,j,0] >= 0:  # 有效點
                    # 投影到原始axis和orthogonal axis
                    axis_component = np.dot(centered_faces[i,j], orig_axis_unit)
                    ortho_component = np.dot(centered_faces[i,j], orig_ortho_unit)

                    # 沿著各自方向縮放
                    scaled_axis_component = axis_component * axis_ratio
                    scaled_ortho_component = ortho_component * ortho_ratio

                    # 重構縮放後的點 (使用目標軸方向)
                    results_vis[0]['faces'][i,j] = (
                        scaled_axis_component * target_axis_unit +
                        scaled_ortho_component * target_ortho_unit +
                        target_center
                    )

        # ref_face_aspect_ratio = compute_face_aspect_ratio(ref_faces[0])
        # # get orign face aspect ratio
        # target_face_aspect_ratio = compute_face_aspect_ratio(results_vis[0]['faces'][0])

        # # print(f"ref_face_aspect_ratio: {ref_face_aspect_ratio}, target_face_aspect_ratio: {target_face_aspect_ratio}")


    face_align_params = {
        'axis_ratio': axis_ratio,
        'ortho_ratio': ortho_ratio,
        'target_center': target_center,
        'orig_center': orig_center,
        'orig_axis_unit': orig_axis_unit,
        'orig_ortho_unit': orig_ortho_unit,
        'target_axis_unit': target_axis_unit,
        'target_ortho_unit': target_ortho_unit,
        'neck_ratio': neck_ratio,
        'shoulder2_ratio': shoulder2_ratio,
        'shoulder5_ratio': shoulder5_ratio,
        'arm3_ratio': arm3_ratio,
        'arm4_ratio': arm4_ratio,
        'arm6_ratio': arm6_ratio,
        'arm7_ratio': arm7_ratio,
        'head14_ratio': head14_ratio,
        'head15_ratio': head15_ratio,
        'head16_ratio': head16_ratio,
        'head17_ratio': head17_ratio,
        'll1_ratio': ll1_ratio,
        'll2_ratio': ll2_ratio,
        'rl1_ratio': rl1_ratio,
        'rl2_ratio': rl2_ratio,
        'x_ratio': x_ratio,
        'y_ratio': y_ratio,
        'offset': offset,
    }
    return results_vis, face_align_params





class SkeletonGeneratorAlign:
    """
    Wraps DWposeDetector to generate skeleton images (with/without face).

    Example
    -------
    >>> sg = SkeletonGenerator()
    >>> sk_with_face, sk_without_face = sg(image)
    """

    def __init__(self, drop_head=False, drop_face=False):
        # 初始化姿态检测模型（一次即可复用）
        self._detector = DWposeDetector()
        self.drop_head = drop_head
        self.drop_face = drop_face
    def __call__(self, ref_image, image, first_frame=False, previous_params=None, use_align=True):
        """
        Generate skeletons for a single image.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image of shape (H, W, 3).

        Returns
        -------
        skeleton_with_face : np.ndarray
            Skeleton image containing face keypoints.
        skeleton_without_face : np.ndarray
            Skeleton image without face keypoints.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("`image` must be a BGR ndarray of shape (H, W, 3).")

        # H, W, _ = image.shape
        H, W = 768, 512

        # 推理姿态
        pose_ref = self._detector(ref_image)
        pose = self._detector(image)

        skeleton_ref = draw_poses(pose_ref, H, W, self.drop_head, self.drop_face)
        skeleton_orig = draw_poses(pose, H, W, self.drop_head, self.drop_face)


        aligned_pose, face_align_params = align_poses(pose_ref[0], pose, first_frame, previous_params, use_align=use_align)


        # 绘制骨架
        skeleton = draw_poses(aligned_pose, H, W, self.drop_head, self.drop_face)

        skeleton_ref = cv2.resize(skeleton_ref, (image.shape[1], image.shape[0]))
        skeleton_orig = cv2.resize(skeleton_orig, (image.shape[1], image.shape[0]))
        skeleton = cv2.resize(skeleton, (image.shape[1], image.shape[0]))

        return skeleton, face_align_params, skeleton_ref, skeleton_orig


def example_usage():
    ref_image = cv2.imread("./pose_debug/input_img_padded.png")
    image = cv2.imread("./pose_debug/render_wo_hair_0001.png")
    if image is None:
        print("Could not load image")
        return

    sg = SkeletonGeneratorAlign()
    skeleton, align_params, skeleton_ref, skeleton_orig = sg(ref_image, image, first_frame=True, previous_params=None)

    cv2.imwrite("pose_debug/skeleton.jpg", skeleton)
    cv2.imwrite("pose_debug/skeleton_ref.jpg", skeleton_ref)
    cv2.imwrite("pose_debug/skeleton_orig.jpg", skeleton_orig)


logger = get_logger('dw pose extraction')


if __name__=='__main__':
    # Example usage
    example_usage()
