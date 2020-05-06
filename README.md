<h2> About this project </h2>

This repository contains the base package of the cleanup challenge files. Each of the ROS services developed for this project has been pushed to its appropriate package under the sensible-robots packages for the University of Leeds LASR team.

<h2> Setup  </h2>

In order to run the code, access to the LASR sensible-robots packages is required. A worksheet for setting up TIAGo's container and acquiring the necessary packages is provided in the LASR yammer group. The necessary packages are:
*  lasr_pcl
*  lasr_moveit
*  lasr_object_detection_yolo
*  jeff_segment_objects
*  utilities

<h2> Installation </h2>

Clone the package under tiago_ws/src and make sure your tiago_ws is built. This can be done by navigating to the top level of the workspace and running
~~~
catkin build
~~~

<h2> Usage </h2>

In order to use the project code, the following server nodes must be running:
*  The transform cloud service 
*  The object approach navigation point service
*  The YOLO object detection service 
*  The manipulation service
*  The segmentation service

The launch file `cleanup.launch` provided in this package runs all the above:
~~~
roslaunch cleanup_challenge cleanup.launch
~~~

Then run the base program node:
~~~
rosrun cleanup_challenge cleanup.py
~~~
