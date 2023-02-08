#!/usr/bin/env python3

from typing import Optional, Any

from segmentation import *
import cv2
from torchvision import transforms
import torchvision
import rospy
import torch
import numpy as np
import os
from sensor_msgs.msg import CompressedImage, Image
from duckietown.dtros import DTROS, DTParam, NodeType, TopicType
from dt_class_utils import DTReminder
from turbojpeg import TurboJPEG
from cv_bridge import CvBridge

class Segmentation:
    def __init__(self, model_dir="/code/catkin_ws/src/dt-ros-commons/packages/ros_commons/model/",
                 model_path="edanetlr=0.0007optim=AdamWepoch=33.pth.tar", num_classes=4, height=480, width=640,
                 env_id=0, device="cuda", ):


        cuda_id = 0

        if env_id % 2 == 1:
            cuda_id = 1

        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        self.device = f"cuda:{cuda_id}"
        print("device", self.device)

        self.model = load_EDANet_model(f"{model_dir}{model_path}", num_classes, self.device)

        self.transform = A.Compose(
            [
                A.Resize(height=height, width=width),
                ToTensorV2(),
            ], )

    def observation(self, obs):
        augmentations = self.transform(image=obs)
        obs = augmentations["image"]
        obs = get_predict(0, obs, self.model, self.device)
        obs = obs.cpu()

        obs = np.dstack([obs[0], obs[0], obs[0]])
        print(obs.shape)
        print(obs)

        mask_white = np.where(obs == [1, 1, 1], [255, 255, 255], [0, 0, 0])

        mask_yellow = np.where(obs == [2, 2, 2], [255, 255, 0], [0, 0, 0])

        mask_red = np.where((obs == [3, 3, 3]), [255, 0, 0], [0, 0, 0])
        mask_green = np.where((obs == [4, 4, 4]), [0, 255, 0], [0, 0, 0])

        test = mask_yellow + mask_red + mask_white+mask_green
        image = test.astype(np.uint8)
        return image


class TestNode(DTROS):
    def __init__(self, node_name):
        super().__init__(node_name, node_type=NodeType.PERCEPTION)

        print(os.getcwd())
        self.segm = Segmentation()
        # parameters
        self.publish_freq = DTParam("~publish_freq", -1)

        # utility objects
        self.bridge = CvBridge()
        self.reminder = DTReminder(frequency=self.publish_freq.value)

        # subscribers
        self.sub_img = rospy.Subscriber(
            "~image_in", CompressedImage, self.cb_image, queue_size=1, buff_size="10MB"
        )

        # publishers
        self.pub_img = rospy.Publisher(
            "~image_out",
            Image,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION,
            dt_healthy_freq=self.publish_freq.value,
            dt_help="Raw image",
        )

    def cb_image(self, msg):
        # make sure this matters to somebody

        img = self.bridge.compressed_imgmsg_to_cv2(msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.segm.observation(img)
        # turn 'raw image' into 'raw image message'
        out_msg = self.bridge.cv2_to_imgmsg(img, "rgb8")
        # maintain original header
        out_msg.header = msg.header
        # publish image
        self.pub_img.publish(out_msg)

if __name__ == "__main__":
    node = TestNode("test_node")
    rospy.spin()