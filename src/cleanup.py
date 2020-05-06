#!/usr/bin/python

import rospy
import actionlib
import cv2
import message_filters
from rospy import ServiceException, ROSInterruptException, ROSException

from sensor_msgs.msg import Image, PointCloud2, RegionOfInterest
from geometry_msgs.msg import Vector3, Pose, Point, Quaternion
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from threading import Event, Lock
from cv_bridge import CvBridge, CvBridgeError

# Classes
from search import Search
from navigate import Navigate
from manipulate import Manipulate

class Cleanup():
    def __init__(self):
        # Create clients
        self.play_motion_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        self.move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)

        # Create a list of all the poses around the room so we can iterate through it
        locations_dict = rospy.get_param("/locations")

        self.locations_order = [loc['name'] for loc in locations_dict]
        self.locations = {
            loc['name']: Pose (
                position=Point(**loc['location']['position']),
                orientation=Quaternion(**loc['location']['orientation'])
            )
            for loc in locations_dict
        }

    def goto_location(self, location_index):
        rospy.loginfo('Going to: %s' % self.locations_order[i])

        self.move_base_client.wait_for_server(rospy.Duration(15.0))

        goal = MoveBaseGoal()
        goal.target_pose.header = Header(frame_id="map", stamp=rospy.Time.now())
        goal.target_pose.pose = self.locations[self.locations_order[i]]

        rospy.loginfo('Sending goal location ...')
        self.move_base_client.send_goal(goal) #waits forever
        if self.move_base_client.wait_for_result():
            rospy.loginfo('Goal location achieved!')
        else:
            rospy.logwarn("Couldn't reach the goal!")

if __name__ == '__main__':
    rospy.init_node('cleanup')
    rospy.loginfo('Started cleanup node...')

    # Create instance of Seach and Manipulate
    cleanup = Cleanup()
    search = Search()
    navigate = Navigate()
    manipulate = Manipulate()

    # Goto each pose
    for i, cleanup_location in enumerate(cleanup.locations):
        # Goto cleanup location
        cleanup.goto_location(i)

        # Search for garbage to cleanup
        garbage = search.search_for_garbage()

        # For each candidate garbage item calculate a navigation approach
        rospy.loginfo('length of garbage is ')
        rospy.loginfo(len(garbage))
        if len(garbage) > 0:
            nav_goals = []
            map_points = []
            motions = []
            for item in garbage:
                nav_goal, garbage_map_point, confirmation_motion = navigate.calculate_approach(item)
                nav_goals.append(nav_goal)
                map_points.append(garbage_map_point)
                motions.append(confirmation_motion)
            # nav_goal, garbage_map_point, confirmation_motion = navigate.calculate_approach(garbage[0])

            for nav_goal, garbage_map_point, confirmation_motion in zip(nav_goals, map_points, motions):
                if nav_goal is not None:
                    # Navigate to candidate item and confirm it is garbage
                    navigate.goto_pose(nav_goal)
                else:
                    continue

                # Send goal again to assure that position is best as possible
                # since orientation can be quite off if nav goal was tight
                navigate.confirm_orientation(garbage_map_point)

                # Get in pose for confirmation
                search.playMotion(confirmation_motion)

                # Confirm its garbage and get a closer up scan
                obj = search.confirm_garbage(item, garbage_map_point)
                if obj is not None:
                    # Look around to build the Octomap of the environment for collision avoidance
                    search.playMotion("head_look_around")

                    # Play the assigned pose to pickup the item, only floor objects have a specific one
                    if confirmation_motion == "low_confirmation_mode":
                        search.playMotion("low_pickup_mode")
                        success = manipulate.pickup(obj, "pickupfloor")
                    else:
                        success = manipulate.pickup(obj, "pickup")

                    # If pickup successful Hold object pose, otherwise maybe create a backup procedure
                    if success:
                        rospy.loginfo('Playing holding object motion')
                        # search.playMotion("holding_object", skip_planning=False) # sometimes he drops the obj while getting into this pose in simulation because of physics 
                        search.playMotion("back_to_default")

                        # Goto bin
                        navigate.goto_place('Bin')
                        search.playMotion("look_bin")

                        # Note: the bin does not segment nicely which results in disposing fail, need to improve
                        # Segment bin and dispose of the picked up garbage
                        bin_info = search.segment_bin() # this function is currently commented out
                        manipulate.dispose(obj, bin_info)

                        # Home pose and move on to the next object found
                        search.playMotion("home", skip_planning=False)
                    else:
                        pass
                        # tts could not find a plan to pickup the object, cant since in sim now
                else:
                    pass
                    # move on to the next object, leaving this here for the comments


