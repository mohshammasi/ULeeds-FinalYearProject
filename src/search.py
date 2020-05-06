#!/usr/bin/python

import rospy
import actionlib
import cv2
import message_filters
import numpy as np
import matplotlib.pyplot as plt
from rospy import ServiceException, ROSInterruptException, ROSException

# Srvs
from lasr_object_detection_yolo.srv import YoloDetection, YoloDetectionResponse, YoloDetectionRequest
from jeff_segment_objects.srv import SegmentObjects, SegmentObjectsRequest, SegmentObjectsResponse
from lasr_pcl.srv import TransformCloudRequest, TransformCloudResponse, TransformCloud
from lasr_pcl.srv import CylinderFromCloud, CylinderFromCloudRequest, CylinderFromCloudResponse

# Msgs
from sensor_msgs.msg import Image, PointCloud2, RegionOfInterest
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal

from threading import Event, Lock
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import KMeans
from math import sqrt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000, delta_e_cie1976, delta_e_cie1994
from collections import namedtuple

# Global class
from utilities import TheGlobalClass

class Search():
    def __init__(self):
        # Create clients
        self.play_motion_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)

        # Services
        self.yolo_detection = rospy.ServiceProxy('/yolo_detection', YoloDetection)
        self.segment_objects_srv = rospy.ServiceProxy('segment_objects', SegmentObjects)
        self.transform_cloud = rospy.ServiceProxy('transform_cloud', TransformCloud)
        self.cylinder_from_cloud = rospy.ServiceProxy('cylinder_from_cloud', CylinderFromCloud)

        # Used for image and pointcloud syncing, 1 message function
        self.image = Image()
        self.cloud = PointCloud2()
        self.mutex = Lock()
        self.cb_event = Event()

        # Bridge
        self.bridge = CvBridge()

        # Callback counter - for DEBUG only
        self.counter = 0

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

    def detect_objects(self, image_msg, dataset, confidence=0.5, nms=0.3):
        try:
            rospy.loginfo('waiting for {}'.format('/yolo_detection'))
            rospy.wait_for_service('/yolo_detection')
            rospy.loginfo('connected to {}'.format('/yolo_detection'))
            detect_req = YoloDetectionRequest(image_raw=image_msg, dataset=dataset, confidence=confidence, nms=nms)
            detect_res = self.yolo_detection(detect_req)
            return detect_res
        except ROSInterruptException:
            pass
        except ServiceException as ex:
            rospy.logwarn('service call failed to {}!'.format('/yolo_detection'))
            rospy.logwarn(ex)
        except ROSException as ex:
            rospy.logwarn('timed out waiting for {}!'.format('/yolo_detection'))
            rospy.logwarn(ex)

    def segment_objects(self, cloud_msg, non_planar_ratio, cluster_tol, min_size, max_size):
        '''
        Segments objects in the pointcloud by clustering.

        arguments:
        non_planar_ratio - controls how much of the cloud should be filtered out.
        cluster_tol      - distance between each point to be considered in 1 cluster e.g 0.02 is 2cm
        min_size         - minimum number of points in a cluster
        max_size         - maximum number of points in a cluster

        returns:
        Segmentation service response.
        '''
        try:
            rospy.loginfo('waiting for {}'.format('/segment_objects'))
            rospy.wait_for_service('segment_objects')
            rospy.loginfo('connected to {}'.format('/segment_objects'))
            segmentation_req = SegmentObjectsRequest(cloud_msg, non_planar_ratio, cluster_tol, min_size, max_size)
            segmentation_res = self.segment_objects_srv(segmentation_req)

            width = cloud_msg.width
            height = cloud_msg.height

            bounding_boxes = []
            for cluster in segmentation_res.clusters:
                max_index = cluster.indices[0]
                left = cluster.indices[0]%width
                right = cluster.indices[0]%width
                bottom = int(cluster.indices[0]/width)
                top = int(cluster.indices[0]/width)

                count = 0 
                for index in cluster.indices:
                    row = int(index/width)
                    col = index%width
                    left = min(left, col)
                    right = max(right, col)
                    top = min(top, row)
                    bottom = max(bottom, row)

                bounding_box = RegionOfInterest()
                bounding_box.x_offset = left
                bounding_box.y_offset = top
                bounding_box.height = bottom - top
                bounding_box.width = right - left
                bounding_boxes.append(bounding_box)
            return bounding_boxes, segmentation_res.objects, segmentation_res.whole # whole cloud is for visualisation DEBUG only
        except rospy.ServiceException as e:
            print e

    def color_filter(self, bounding_box, image):
        margin = 5

        # Get bigger bounding box around the original bounding box
        x1 = bounding_box.x_offset - margin 
        y1 = bounding_box.y_offset - margin
        x2 = bounding_box.x_offset + bounding_box.width + margin
        y2 = bounding_box.y_offset + bounding_box.height + margin

        # Ensure that points are in the image
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 >= 640:
            x2 = 639
        if y2 >= 480:
            y2 = 479
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Loop through new bigger bounding box and only pick pixels around original bb
        background = []
        foreground = []
        temp = y1
        while (x1 < x2):
            while (y1 < y2):
                # Check that the pixel is not in the original bounding box
                if ( ( bounding_box.y_offset <= y1 <= bounding_box.y_offset+bounding_box.height ) and ( bounding_box.x_offset <= x1 <= bounding_box.x_offset+bounding_box.width ) ):
                    foreground.append(image[y1, x1])
                else:
                    background.append(image[y1, x1])
                y1 = y1 + 1
            y1 = temp
            x1 = x1 + 1

        background = np.array(background)
        foreground = np.array(foreground)

        # cluster the pixel intensity
        bg_cluster = KMeans(n_clusters=1)
        bg_cluster.fit(background)

        bg_rgb = bg_cluster.cluster_centers_[0]
        bg_rgb = sRGBColor(bg_rgb[0]/255, bg_rgb[1]/255, bg_rgb[2]/255) # range 0-1
        # rospy.loginfo('BG in RBG is ')
        # rospy.loginfo(bg_rgb)
        bg_lab = convert_color(bg_rgb, LabColor)
        # rospy.loginfo('BG in LAB is ')
        # rospy.loginfo(bg_lab)

        # cluster the pixel intensity
        fg_cluster = KMeans(n_clusters=1)
        fg_cluster.fit(foreground)

        fg_rgb = fg_cluster.cluster_centers_[0]
        fg_rgb = sRGBColor(fg_rgb[0]/255, fg_rgb[1]/255, fg_rgb[2]/255)
        # rospy.loginfo('fG in RBG is ')
        # rospy.loginfo(fg_rgb)
        fg_lab = convert_color(fg_rgb, LabColor)
        # rospy.loginfo('fG in LAB is ')
        # rospy.loginfo(fg_lab)

        # Find the color difference
        delta_e = delta_e_cie2000(fg_lab, bg_lab)
        # delta_e = delta_e_cie1994(fg_lab, bg_lab)
        print "The difference between the 2 color = ", delta_e

        # # DEBUG VISUALISATION
        # # build a histogram of the cluster and then create a figure to see the primary color
        # bg_hist = self.centroid_histogram(bg_cluster)
        # bg_bar = self.plot_colors(bg_hist, bg_cluster.cluster_centers_)
        # # build a histogram of the cluster and then create a figure to see the primary color
        # fg_hist = self.centroid_histogram(fg_cluster)
        # fg_bar = self.plot_colors(fg_hist, fg_cluster.cluster_centers_)

        # # show our color bart
        # plt.figure(num='Background primary color')
        # plt.imshow(bg_bar)
        # # plt.axis("off")
        # plt.figure(num='Foreground primary color')
        # plt.imshow(fg_bar)
        # plt.show()

        if delta_e >= 10:
            return True, fg_lab
        else:
            return False, 0

    # # CAN DELETE, USED ONLY FOR VISUALISATION AND DEBUG
    # def centroid_histogram(self, clt):
    #     # grab the number of different clusters and create a histogram
    #     # based on the number of pixels assigned to each cluster
    #     numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    #     (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    #     # normalize the histogram, such that it sums to one
    #     hist = hist.astype("float")
    #     hist /= hist.sum()

    #     return hist

    # # CAN DELETE, USED ONLY FOR VISUALISATION AND DEBUG
    # def plot_colors(self, hist, centroids):
    #     # initialize the bar chart representing the relative frequency
    #     # of each of the colors
    #     bar = np.zeros((50, 300, 3), dtype = "uint8")
    #     startX = 0

    #     # loop over the percentage of each cluster and the color of
    #     # each cluster
    #     for (percent, color) in zip(hist, centroids):
    #         # plot the relative percentage of each cluster
    #         endX = startX + (percent * 300)
    #         cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
    #         startX = endX
        
    #     return bar

    def dimensions_filter(self, item):
        big = 0.25
        # Check that none of the dimensions is absolutely massive
        if item.width > big or item.height > big or item.depth > big:
            return True
        
        tiny = 0.01
        # Check that none of the dimensions is super tiny 
        if item.width < tiny or item.height < tiny or item.depth < tiny:
            return True

        # Populate this filter with more testing when finding irregular cases
        
        return False

    # Tutorial reference https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # Used to remove valuable detected items from the list of garbage items
    def bb_intersection_over_union(self, prediction_box, detections):
        for detection in detections:
            # determine the (x, y)-coordinates of the intersection rectangle
            x1 = max(prediction_box.x_offset, detection.xywh[0])
            y1 = max(prediction_box.y_offset, detection.xywh[1])
            x2 = min(prediction_box.x_offset+prediction_box.width, detection.xywh[0]+detection.xywh[2])
            y2 = min(prediction_box.y_offset+prediction_box.height, detection.xywh[1]+detection.xywh[3])

            # compute the area of intersection rectangle
            intersectionArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

            # compute the area of both the bounding boxes
            box1Area = (prediction_box.x_offset+prediction_box.width - prediction_box.x_offset + 1) * (prediction_box.y_offset+prediction_box.height - prediction_box.y_offset + 1)
            box2Area = (detection.xywh[0]+detection.xywh[2] - detection.xywh[0] + 1) * (detection.xywh[1]+detection.xywh[3] - detection.xywh[1] + 1)

            # compute the intersection over union 
            iou = intersectionArea / float(box1Area + box2Area - intersectionArea)

            # Check score
            if iou >= 0.5:
                return True

        return False

    def search_for_garbage(self):
        # Make sure his head is in the correct position
        self.playMotion("cleanup_look_down")

        # Get image and cloud and run both detection and segmentation on image and cloud
        self.get_image_and_cloud()

        # Call detection on the image
        detection_res = self.detect_objects(self.image, "coco", 0.1, 0.3)

        # Call segmentation on the pointcloud
        bounding_boxes, objects, whole_cloud = self.segment_objects(self.cloud, 0.25, 0.03, 200, 2000)

        # PUBLISH DEBUG CLOUD
        pub = rospy.Publisher('wholeting', PointCloud2, queue_size=10)
        rate = rospy.Rate(10) # 10hz
        for i in range(10):
            pub.publish(whole_cloud)
            rate.sleep()

        # # extra images for testing to see the difference in bounding boxes
        # try:
        #     frame = self.bridge.imgmsg_to_cv2(detection_res.image_bb, 'bgr8')
        #     test_frame0 = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
        #     test_frame1 = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
        #     test_frame2 = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
        #     test_frame3 = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
        # except CvBridgeError:
        #     return

        # # label the first test image with all the bounding boxes found from both sources 
        # for detection in detection_res.detected_objects:
        #     cv2.rectangle(test_frame0, (int(detection.xywh[0]), int(detection.xywh[1])), (int(detection.xywh[0]+detection.xywh[2]), int(detection.xywh[1]+detection.xywh[3])), (255, 0, 255), 1)

        # for bounding_box in bounding_boxes:
        #     cv2.rectangle(test_frame0, (bounding_box.x_offset, bounding_box.y_offset), (bounding_box.x_offset+bounding_box.width, bounding_box.y_offset+bounding_box.height), (255, 255, 0), 1)


        # Apply the color filter on the bounding boxes
        i = 0
        n = len(bounding_boxes)
        colors = []
        while i < n:
            foundDifference, fg_color = self.color_filter(bounding_boxes[i], frame)

            # If there is no color difference its 'probably' noise and remove it from both lists
            if not foundDifference:
                bounding_boxes.pop(i)
                objects.pop(i)
                n = n - 1
            else:
                i = i + 1
                colors.append(fg_color)
        rospy.loginfo('Color filter applied!')
        
        # # label images with bb after colour filter
        # for bounding_box in bounding_boxes:
        #     cv2.rectangle(test_frame1, (bounding_box.x_offset, bounding_box.y_offset), (bounding_box.x_offset+bounding_box.width, bounding_box.y_offset+bounding_box.height), (255, 255, 0), 1)

        # Apply the dimension filter on the remaining objects
        i = 0
        n = len(objects)
        while i < n:
            foundBad = self.dimensions_filter(objects[i])

            if foundBad:
                bounding_boxes.pop(i)
                objects.pop(i)
                colors.pop(i)
                n = n - 1
            else:
                i = i + 1
        rospy.loginfo('Dimensions filter applied!')

        # # label images after 2 filters
        # for bounding_box in bounding_boxes:
        #     cv2.rectangle(test_frame2, (bounding_box.x_offset, bounding_box.y_offset), (bounding_box.x_offset+bounding_box.width, bounding_box.y_offset+bounding_box.height), (255, 255, 0), 1)
        # for bounding_box in bounding_boxes:
        #     cv2.rectangle(frame, (bounding_box.x_offset, bounding_box.y_offset), (bounding_box.x_offset+bounding_box.width, bounding_box.y_offset+bounding_box.height), (255, 255, 0), 1)

        # # Remove anything that has a huge bounding box similarity with a detected 'valuable' item coz in sim now
        # # This is part of the MAIN task, only commented out because there is no specific 'valuables' set in simulation
        # i = 0
        # n = len(bounding_boxes)
        # while i < n:
        #     foundHighScore = self.bb_intersection_over_union(bounding_boxes[i], detection_res.detected_objects)

        #     # If we found a high score then its probably not garbage, remove from garbage list
        #     if foundHighScore:
        #         bounding_boxes.pop(i)
        #         objects.pop(i)
        #         colors.pop(i)
        #         n = n - 1
        #     else:
        #         i = i + 1
        
        # # Label image with bbs after all filters
        # for bounding_box in bounding_boxes:
        #     cv2.rectangle(test_frame3, (bounding_box.x_offset, bounding_box.y_offset), (bounding_box.x_offset+bounding_box.width, bounding_box.y_offset+bounding_box.height), (255, 255, 0), 1)

        # cv2.imshow('Result', test_frame0)
        # cv2.imshow('After COLOR filter', test_frame1)
        # cv2.imshow('After DIMENSIONS filter', frame)
        # cv2.imshow('Possible Garbage objects', test_frame3)
        # cv2.waitKey(0)
                
        # return list of candidate garbage objects found
        return objects

        #########################################################################################
        # # This is for multiple detections when looking around, failed miserably -> detection_and_segmentation_callback function
        # # Play the look around motion to search
        # self.playMotion("head_look_around")

        # # Subscribe to both the pointcloud and image and run detection and segmentation
        # # while looking around
        # self.cloud_sub = message_filters.Subscriber('/xtion/depth_registered/points', PointCloud2)
        # self.image_sub = message_filters.Subscriber('/xtion/rgb/image_rect_color', Image)
        # ts = message_filters.ApproximateTimeSynchronizer([self.cloud_sub, self.image_sub], 1, 0.1)
        # ts.registerCallback(self.detection_and_segmentation_callback)
        # rospy.spin()
        #########################################################################################

    def get_image_and_cloud(self):
        '''
        Fetches and sync's the RGB image and Pointcloud, only 1 message from each.

        arguments:
        No arguments needed.

        returns:
        The RGB image and PointCloud (As ROS messages).
        '''
        rospy.loginfo('Getting image and cloud...')
        self.cb_event = Event()

        # Our Subscribers that we would like to sync
        image_sub = message_filters.Subscriber('/xtion/rgb/image_rect_color', Image)
        cloud_sub = message_filters.Subscriber('/xtion/depth_registered/points', PointCloud2)

        # Sync the subscribers messages and assign them in callback
        sync = message_filters.ApproximateTimeSynchronizer([image_sub, cloud_sub], 10, 0.1)
        sync.registerCallback(self.single_message_callback)
        rospy.loginfo('registerCallback finished...')

        # Wait for event to be set and lock freed
        while not rospy.is_shutdown() and not self.cb_event.is_set():
            rospy.loginfo('sleeping...')
            rospy.sleep(0.1)
        rospy.loginfo('after event...')

        image_sub.sub.unregister()
        cloud_sub.sub.unregister()

        # Transform the cloud to be in base_footprint
        rospy.loginfo('waiting for {}'.format('/transform_cloud'))
        rospy.wait_for_service('transform_cloud')
        rospy.loginfo('connected to {}'.format('/transform_cloud'))
        transform_req = TransformCloudRequest(self.cloud, "base_footprint")
        transform_res = self.transform_cloud(transform_req)
        self.cloud = transform_res.transformed_cloud
        rospy.loginfo('The self.cloud is now in base footprint!')

    def single_message_callback(self, image, cloud):
        '''
        The callback syncing RGB image and PointCloud.

        arguments:
        image   - The rgb image message gotten by the subscriber. 
        cloud   - The Pointcloud message gotten by the subscriber.

        returns:
        Nothing.
        '''
        self.mutex.acquire()
        try:
            rospy.logwarn('Sync cloud and image callback called... {}'.format(rospy.Time.now() - cloud.header.stamp < rospy.Duration(0.5)))
            if not self.cb_event.is_set() and rospy.Time.now() - cloud.header.stamp < rospy.Duration(0.5):
                rospy.logwarn('Assigning image and cloud...')
                self.image = image
                self.cloud = cloud
                rospy.loginfo('Setting event...')
                self.cb_event.set()
        finally:
            self.mutex.release()

    # # Failed Experiement - not so good results
    # def color_difference(self, original_color, bounding_box, image):
    #     x1 = bounding_box.x_offset
    #     y1 = bounding_box.y_offset
    #     x2 = bounding_box.x_offset+bounding_box.width
    #     y2 = bounding_box.y_offset+bounding_box.height
    #     foreground = []
    #     temp = y1
    #     while (x1 < x2):
    #         while (y1 < y2):
    #             # gather fg pixels
    #             foreground.append(image[y1, x1])
    #             y1 = y1 + 1
    #         y1 = temp
    #         x1 = x1 + 1

    #     foreground = np.array(foreground)

    #     # cluster the pixel intensity
    #     fg_cluster = KMeans(n_clusters=1)
    #     fg_cluster.fit(foreground)

    #     fg_rgb = fg_cluster.cluster_centers_[0]
    #     fg_rgb = sRGBColor(fg_rgb[0]/255, fg_rgb[1]/255, fg_rgb[2]/255)
    #     fg_lab = convert_color(fg_rgb, LabColor)
    #     # rospy.loginfo('fG in LAB is ')
    #     # rospy.loginfo(fg_lab)

    #     # Find the color difference
    #     delta_e = delta_e_cie2000(original_color, fg_lab)
    #     print "cd-The difference between the 2 color = ", delta_e

    #     if delta_e >= 4:
    #         return True
    #     else:
    #         return False

    # I am confirming the object by comparing the previously calculated position of the object on the map with the
    # new calculated position of the object on the map. This is not so great as the further away the object the greater
    # the error when calculating its position point from far away.
    def confirm_garbage(self, garbage, map_point):
        # Get image and cloud and run both detection and segmentation on image and cloud
        self.get_image_and_cloud()

        # Call detection on the image
        detection_res = self.detect_objects(self.image, "coco", 0.7, 0.3)

        # Call segmentation on the pointcloud
        # bounding_boxes, objects, whole_cloud = self.segment_objects(self.cloud, 0.6, 0.03, 500, 7500)
        bounding_boxes, objects, whole_cloud = self.segment_objects(self.cloud, 0.2, 0.03, 200, 9500)

        # Remove anything that has a huge bounding box similarity with a detected 'valuable' item i.e. overlap 
        i = 0
        n = len(bounding_boxes)
        while i < n:
            foundHighScore = self.bb_intersection_over_union(bounding_boxes[i], detection_res.detected_objects)

            # If we found a high score then its probably not garbage, remove from garbage list
            if foundHighScore:
                bounding_boxes.pop(i)
                objects.pop(i)
                n = n - 1
            else:
                i = i + 1

        if len(bounding_boxes) == 0:
            rospy.logwarn('Everything filtered :(')

        # DEBUG FOR NOW
        pub = rospy.Publisher('wholeting', PointCloud2, queue_size=10)
        rate = rospy.Rate(10) # 10hz
        for i in range(10):
            pub.publish(whole_cloud)
            rate.sleep()

        # To know exactly which garbage item we came for we calculate map point
        # for each thing and distance from the original map point
        pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped).pose.pose
        i = 0
        n = len(bounding_boxes)
        while i < n:
            # Calculate the map point of the object
            # Put object center in Point()
            p = Point()
            p.x = objects[i].center.pose.position.x
            p.y = objects[i].center.pose.position.y

            # Transform the object point to map frame
            item_map = TheGlobalClass.base2map(p, pose)

            # Calculate distance between the two points
            x_diff = map_point.x-item_map.x
            y_diff = map_point.y-item_map.y
            distance = sqrt((x_diff*x_diff) + (y_diff*y_diff))
            rospy.loginfo('distance is ')
            rospy.loginfo(distance)
            if distance > 0.085: # threshold, can experiement with this
                objects.pop(i)
                n = n - 1
            else:
                i = i + 1

        rospy.loginfo('whats left is ')
        rospy.loginfo(len(objects))

        # Image to see the result of the confirmation, can remove
        try:
            test_frame0 = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
        except CvBridgeError:
            return

        for bounding_box in bounding_boxes:
            cv2.rectangle(test_frame0, (bounding_box.x_offset, bounding_box.y_offset), (bounding_box.x_offset+bounding_box.width, bounding_box.y_offset+bounding_box.height), (255, 255, 0), 1)

        # label the image with all the bounding boxes found from both sources
        for detection in detection_res.detected_objects:
            cv2.rectangle(test_frame0, (int(detection.xywh[0]), int(detection.xywh[1])), (int(detection.xywh[0]+detection.xywh[2]), int(detection.xywh[1]+detection.xywh[3])), (255, 0, 255), 1)

        cv2.imshow('Garbage confirmation', test_frame0)
        cv2.waitKey(0)

        if len(objects) == 1:
            return objects[0]
        else:
            return None

    # Must remember that most the services expect that the cloud is already in base_footprint
    # Doesnt perform as well as expected, cant segment the bin nicely
    def segment_bin(self):
        # # Get image and cloud and run cylinder segmentation to get bin
        # self.get_image_and_cloud()

        # try:
        #     rospy.loginfo('waiting for {}'.format('/cylinder_from_cloud'))
        #     rospy.wait_for_service('cylinder_from_cloud')
        #     rospy.loginfo('connected to {}'.format('/cylinder_from_cloud'))
        #     bin_req = CylinderFromCloudRequest(self.cloud, 0, 100, 0, 100) # basically filter nothing
        #     bin_res = self.cylinder_from_cloud(bin_req)
        # except rospy.ServiceException as e:
        #     print e
        # return bin_res
        pass



