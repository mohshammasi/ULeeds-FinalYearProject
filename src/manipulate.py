#!/usr/bin/env python

import rospy
import actionlib
import cv2

from math import sqrt

# Srvs
from rospy import ServiceException, ROSInterruptException, ROSException
from lasr_pcl.srv import TransformCloudRequest, TransformCloudResponse, TransformCloud
from lasr_moveit.srv import ManipulationRequest, ManipulationResponse, Manipulation

# Msgs
from geometry_msgs.msg import Pose, Quaternion, Point, PoseWithCovarianceStamped, Vector3, Twist
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal



class Manipulate():
    def __init__(self):
        self.play_motion_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        # Services
        self.manipulation_service = rospy.ServiceProxy('manipulation', Manipulation)
        self.base_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)

    def playMotion(self, motion_name, skip_planning=True, blocking=True):
        # Wait for the play motion server to come up and send goal
        self.play_motion_client.wait_for_server(rospy.Duration(15.0))

        # Create the play_motion goal and send it
        pose_goal = PlayMotionGoal()
        pose_goal.motion_name = motion_name
        pose_goal.skip_planning = skip_planning # Must be True for motions without arm motion
        self.play_motion_client.send_goal(pose_goal)
        rospy.loginfo('Play motion goal sent')
        if blocking:
            self.play_motion_client.wait_for_result()

    def pickup(self, obj, action):
        try:
            rospy.loginfo('waiting for {}'.format('/manipulation'))
            rospy.wait_for_service('manipulation')
            rospy.loginfo('connected to {}'.format('/manipulation'))
            manipulation_req = ManipulationRequest(obj=obj, command=action, planning_attempts=50)
            manipulation_res = self.manipulation_service(manipulation_req)
            rospy.loginfo('Manipulation resposne is ')
            rospy.loginfo(manipulation_res.success)
        except rospy.ServiceException as e:
            print e
        return manipulation_res.success

    # The commented out parts of the code is because the segment bin function in search.py is commented out
    def dispose(self, obj, bin_info):
        try:
            # plan to exactly above the bin in center
            # margin = 0.1
            # bin_info.cylinder_center.position.z = bin_info.cylinder_center.position.z + bin_info.cylinder_height/2 + margin
            rospy.loginfo('waiting for {}'.format('/manipulation'))
            rospy.wait_for_service('manipulation')
            rospy.loginfo('connected to {}'.format('/manipulation'))
            # manipulation_req = ManipulationRequest(goal_pose=bin_info.cylinder_center, command="place", planning_attempts=10)
            manipulation_req = ManipulationRequest(obj=obj, command="place", planning_attempts=10)
            manipulation_res = self.manipulation_service(manipulation_req)
            rospy.loginfo('Manipulation resposne is ')
            rospy.loginfo(manipulation_res.success)
            rospy.sleep(2)
        except rospy.ServiceException as e:
            print e
        return manipulation_res.success