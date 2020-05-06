#!/usr/bin/env python

import rospy
import actionlib
import cv2

from math import sqrt

# Srvs
from rospy import ServiceException, ROSInterruptException, ROSException
from lasr_pcl.srv import TransformCloudRequest, TransformCloudResponse, TransformCloud
from lasr_pcl.srv import ObjectApproachNavPointRequest, ObjectApproachNavPointResponse, ObjectApproachNavPoint

# Msgs
from geometry_msgs.msg import Pose, Quaternion, Point, PoseWithCovarianceStamped, Vector3
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# Global class
from utilities import TheGlobalClass


class Navigate():
    def __init__(self):
        # Create clients
        self.play_motion_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        self.move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)

        # Services
        self.transform_cloud = rospy.ServiceProxy('transform_cloud', TransformCloud)
        self.object_approach_nav_point_srv = rospy.ServiceProxy('object_approach_nav_point', ObjectApproachNavPoint)

        # Initialise stuff
        self.cloud = PointCloud2()

    def goto_pose(self, goal_pose):
        self.move_base_client.wait_for_server(rospy.Duration(15.0))

        goal = MoveBaseGoal()
        goal.target_pose.header = Header(frame_id="map", stamp=rospy.Time.now())
        goal.target_pose.pose = goal_pose

        rospy.loginfo('Sending goal location ...')
        self.move_base_client.send_goal(goal) #waits forever
        if self.move_base_client.wait_for_result():
            rospy.loginfo('Goal location achieved!')
        else:
            rospy.logwarn("Couldn't reach the goal!")
    
    def goto_place(self, place):
        self.move_base_client.wait_for_server(rospy.Duration(15.0))
        place_info = rospy.get_param('/' + place)

        goal = MoveBaseGoal()
        goal.target_pose.header = Header(frame_id="map", stamp=rospy.Time.now())
        goal.target_pose.pose = Pose(position=Point(**place_info['location']['position']),
            orientation=Quaternion(**place_info['location']['orientation']))

        rospy.loginfo('Sending goal location ...')
        self.move_base_client.send_goal(goal) #waits forever
        if self.move_base_client.wait_for_result():
            rospy.loginfo('Goal location achieved!')
        else:
            rospy.logwarn("Couldn't reach the goal!")

    def get_cloud(self, frame_id):
        cloud = rospy.wait_for_message('/xtion/depth_registered/points', PointCloud2)

        # Transform the cloud to be in base_footprint
        if frame_id != "xtion_rgb_optical_frame":
            try:
                rospy.loginfo('waiting for {}'.format('/transform_cloud'))
                rospy.wait_for_service('transform_cloud')
                rospy.loginfo('connected to {}'.format('/transform_cloud'))
                transform_req = TransformCloudRequest(cloud, frame_id)
                transform_res = self.transform_cloud(transform_req)
                self.cloud = transform_res.transformed_cloud
                rospy.loginfo('The cloud is now in ' + frame_id)
            except rospy.ServiceException as e:
                print e

    def calculate_approach(self, item):
        # Put object center in Point()
        center = Point()
        center.x = item.center.pose.position.x
        center.y = item.center.pose.position.y

        # Turn tiago towards the object just to get a better view of everything around it
        pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped).pose.pose
        # Transform the object point to map frame
        item_map = TheGlobalClass.base2map(center, pose)
        quaternion = TheGlobalClass.quaternion_at_point(item_map, pose)
        pose.orientation = quaternion

        # self.goto_pose(pose) # face towards the object 
        # disabled facing coz it messes up the nav point calculation for other objects. changed approach

        # Since tiago moved slightly the previous point in base_footprint is invalid
        item_base = TheGlobalClass.map2base(item_map, pose)

        # Check if the garbage is on the floor then the approach goal is just slightly infront of it 
        # No need to do the plane calculations
        rospy.loginfo('Garbage level from ground: ')
        rospy.loginfo(item.center.pose.position.z-item.height/2)
        distance_away = 0.4
        if item.center.pose.position.z < 0.15:
            # Take the object point and get a point some distance infront of it
            base_in_map = Point()
            base_in_map.x = pose.position.x
            base_in_map.y = pose.position.y
            base_in_base = TheGlobalClass.map2base(base_in_map, pose)

            direction = Vector3()
            direction.x = base_in_base.x-item.center.pose.position.x
            direction.y = base_in_base.y-item.center.pose.position.y
            norm = sqrt((direction.x*direction.x) + (direction.y*direction.y))

            u = Vector3()
            u.x = distance_away*(direction.x/norm)
            u.y = distance_away*(direction.y/norm)

            nav_point_base = Point()
            nav_point_base.x = item.center.pose.position.x + u.x
            nav_point_base.y = item.center.pose.position.y + u.y

            # Transform the point to the map frame
            nav_point_map = TheGlobalClass.base2map(nav_point_base, pose)

            # Get the quaternion for the nav goal, create pose and return
            quaternion = TheGlobalClass.quaternion_from_point_at_point(nav_point_map, item_map)
            nav_goal = Pose()
            nav_goal.position.x = nav_point_map.x
            nav_goal.position.y = nav_point_map.y
            nav_goal.orientation = quaternion

            # Check point with make plan
            possible, gotten, tol = TheGlobalClass.make_plan(nav_point_map, tol_gap=0.01, current_pose=pose)
            if possible:
                rospy.loginfo('Found a plan! ')
                return nav_goal, item_map, "low_confirmation_mode"
            else:
                rospy.loginfo('Did not find a plan :( ')
                return None
        else: # If object is on some sort of a surface 
            # Call the get approach nav point service 
            margin = 0.03
            try:
                rospy.loginfo('waiting for {}'.format('/object_approach_nav_point'))
                rospy.wait_for_service('object_approach_nav_point')
                rospy.loginfo('connected to {}'.format('/object_approach_nav_point'))
                self.get_cloud("base_footprint")
                approach_nav_point_req = ObjectApproachNavPointRequest(self.cloud, item.center.pose.position.z-item.height/2-margin, item.center.pose.position.z-item.height/2+margin, 0.0, 100.0, item_base)
                approach_nav_point_res = self.object_approach_nav_point_srv(approach_nav_point_req)
                rospy.loginfo('Returned...')
            except rospy.ServiceException as e:
                print e

            # Transform the nav point to map frame and create final nav pose
            nav_goals_map = []
            i = 0
            for point in approach_nav_point_res.nav_points:
                nav_point_map = TheGlobalClass.base2map(point, pose)
                quaternion = TheGlobalClass.quaternion_from_point_at_point(nav_point_map, item_map)
                nav_goal = Pose()
                nav_goal.position.x = nav_point_map.x
                nav_goal.position.y = nav_point_map.y
                nav_goal.orientation = quaternion

                # Check point with make plan
                p = Point()
                p.x = nav_point_map.x
                p.y = nav_point_map.y
                possible, gotten, tol = TheGlobalClass.make_plan(p, tol_gap=0.01, current_pose=pose)
                if possible:
                    rospy.loginfo('Found a plan! ' + str(i) + ' ' + str(tol))
                    nav_goals_map.append(nav_goal)
                else:
                    rospy.loginfo('Did not find a plan :( ' + str(i))
                i = i + 1

            # # DEBUG -- Visualise edge point - can remove this
            # rospy.loginfo('Publishing the points')
            # hull_pub = rospy.Publisher('/gotten_hull', PointCloud2, queue_size=10)
            # hull_pub2 = rospy.Publisher('/gotten_hull2', PointCloud2, queue_size=10)
            # marker_pub = rospy.Publisher('/edge_p', Marker, queue_size=10)
            # marker_pub2 = rospy.Publisher('/map_p', Marker, queue_size=10)
            # rospy.sleep(3)
            
            # if marker_pub2.get_num_connections() > 0:
            #     for i, point in enumerate(approach_nav_point_res.nav_points):
            #         marker = Marker()
            #         marker.header = Header(frame_id="base_footprint", stamp=rospy.Time.now())
            #         marker.ns = 'basic_shapes'
            #         marker.id = i
            #         marker.type = marker.SPHERE
            #         marker.action = marker.ADD
            #         marker.lifetime.secs = 40
            #         marker_pose = Pose()
            #         marker_pose.position.x = point.x
            #         marker_pose.position.y = point.y
            #         marker_pose.orientation.w = 1.0
            #         marker.pose = marker_pose
            #         marker.color.a = 1.0
            #         marker.color.g = 1.0
            #         scale = 0.1
            #         marker.scale.x = scale
            #         marker.scale.y = scale
            #         marker.scale.z = scale
            #         marker_pub.publish(marker)

            #     for i, nav_goal in enumerate(nav_goals_map):
            #         marker = Marker()
            #         marker.header = Header(frame_id="map", stamp=rospy.Time.now())
            #         marker.ns = 'basic_shapes'
            #         marker.id = i
            #         marker.type = marker.SPHERE
            #         marker.action = marker.ADD
            #         marker.lifetime.secs = 40
            #         marker.pose = nav_goal
            #         marker.color.a = 1.0
            #         marker.color.b = 1.0
            #         scale = 0.1
            #         marker.scale.x = scale
            #         marker.scale.y = scale
            #         marker.scale.z = scale
            #         marker_pub2.publish(marker)

            #     rospy.sleep(1)
            # else:
            #     rospy.loginfo('Nope pub not fully setup I guess')

            # for i in range(10):
            #     hull_pub.publish(approach_nav_point_res.hull)
            #     rospy.sleep(0.1)
            # rospy.sleep(3)
            # for i in range(10):
            #     hull_pub2.publish(approach_nav_point_res.hull2)
            #     rospy.sleep(0.1)

            if len(nav_goals_map) > 0:
                return nav_goals_map[0], item_map, "medium_mode" # [0] to always return the closest 'best' one
            else:
                return None    

    def confirm_orientation(self, garbage_point):
        # Turn tiago towards the object
        pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped).pose.pose
        quaternion = TheGlobalClass.quaternion_at_point(garbage_point, pose)
        pose.orientation = quaternion
        self.goto_pose(pose) # face towards the object

