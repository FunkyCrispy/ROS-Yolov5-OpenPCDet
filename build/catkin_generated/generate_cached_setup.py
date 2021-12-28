# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import stat
import sys

# find the import for catkin's python package - either from source space or from an installed underlay
if os.path.exists(os.path.join('/opt/ros/kinetic/share/catkin/cmake', 'catkinConfig.cmake.in')):
    sys.path.insert(0, os.path.join('/opt/ros/kinetic/share/catkin/cmake', '..', 'python'))
try:
    from catkin.environment_cache import generate_environment_script
except ImportError:
    # search for catkin package in all workspaces and prepend to path
    for workspace in '/home/fengchen/projects/ros_python3_env/devel_isolated/vision_opencv;/home/fengchen/projects/ros_python3_env/devel_isolated/tf_conversions;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2_tools;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2_sensor_msgs;/home/fengchen/projects/ros_python3_env/devel_isolated/test_tf2;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2_kdl;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2_geometry_msgs;/home/fengchen/projects/ros_python3_env/devel_isolated/tf;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2_ros;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2_py;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2_eigen;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2_bullet;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2;/home/fengchen/projects/ros_python3_env/devel_isolated/tf2_msgs;/home/fengchen/projects/ros_python3_env/devel_isolated/rostopic;/home/fengchen/projects/ros_python3_env/devel_isolated/rosnode;/home/fengchen/projects/ros_python3_env/devel_isolated/rosmsg;/home/fengchen/projects/ros_python3_env/devel_isolated/message_filters;/home/fengchen/projects/ros_python3_env/devel_isolated/cv_bridge;/home/fengchen/projects/ros_python3_env/devel_isolated/rostest;/home/fengchen/projects/ros_python3_env/devel_isolated/rosservice;/home/fengchen/projects/ros_python3_env/devel_isolated/rospy;/home/fengchen/projects/ros_python3_env/devel_isolated/rosparam;/home/fengchen/projects/ros_python3_env/devel_isolated/rosmaster;/home/fengchen/projects/ros_python3_env/devel_isolated/roslaunch;/home/fengchen/projects/ros_python3_env/devel_isolated/rosgraph;/home/fengchen/projects/ros_python3_env/devel_isolated/opencv_tests;/home/fengchen/projects/ros_python3_env/devel_isolated/kdl_conversions;/home/fengchen/projects/ros_python3_env/devel_isolated/image_geometry;/home/fengchen/projects/ros_python3_env/devel_isolated/geometry2;/home/fengchen/projects/ros_python3_env/devel_isolated/geometry;/home/fengchen/projects/ros_python3_env/devel_isolated/eigen_conversions;/home/fengchen/projects/ros_project/devel;/opt/ros/kinetic'.split(';'):
        python_path = os.path.join(workspace, 'lib/python3/dist-packages')
        if os.path.isdir(os.path.join(python_path, 'catkin')):
            sys.path.insert(0, python_path)
            break
    from catkin.environment_cache import generate_environment_script

code = generate_environment_script('/home/fengchen/projects/ros-yolov5-pointpillar/devel/env.sh')

output_filename = '/home/fengchen/projects/ros-yolov5-pointpillar/build/catkin_generated/setup_cached.sh'
with open(output_filename, 'w') as f:
    # print('Generate script for cached setup "%s"' % output_filename)
    f.write('\n'.join(code))

mode = os.stat(output_filename).st_mode
os.chmod(output_filename, mode | stat.S_IXUSR)
