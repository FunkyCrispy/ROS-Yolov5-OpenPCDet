
# ROS-Yolov5-OpenPCDet

## Introduction
---
This is a multi-sensor fusion (camera, lidar) project based on Yolov5, OpenPCDet and ROS.

Yolov5 and OpenPCDet are modules of this project, thus importing sentences in some scripts in them based on module name are modified. Eg. pcdet is a module of OpenPCDet, some scripts in **'pcdet'** module writing importing sentence like **"from pcdet.xxx import xxx"**, and these sentences are changed to **"from OpenPCDet.pcdet.xxx import xxx"**. 

Also, some other places are modified in order to correctly work. Similarly, some files in yolov5 module also add 'yolov5' in front of module names. 
**For model training process, its suggested to download Yolov5 or OpenPCDet source code and training separately.**

Through ROS's ApproximateTimeSynchronizer in message_filters module, we get synchronized message pair of topic:
```
/rslidar_points  --  lidar
/ue09_pylon02/image_raw  --  camera
```
And publish topic:
```
/sensor_fusion/detection_bbox  --  boundingbox message
/sensor_fusion/detection_img  --  image plotted bboxes
/sensor_fusion/sync_pcl  --  synchronized point cloud for better visualization
```
Now the **total minmal processing time cost around 0.17s**, the yolov5 inference and get semantic image part cost 0.08s, the semantic point cloud generation part cost 0.04s, the OpenPCDet model PointPillar cost 0.02s, and topic message synchronization and publishing cost the rest of time.

Further acceleration is necessary.

## Module's Source Codes
```
Yolov5 -- git clone https://github.com/ultralytics/yolov5.git
OpenPCDet -- git clone https://github.com/open-mmlab/OpenPCDet.git
```
## Procedure
1. Image detection based on Yolov5
   Produce semantic image, in which colored rectangle are plotted based on the detection boundingboxes.
2. Point cloud points projection
   Point cloud points are projected to image plane ([x, y, z] --> [u, v]), and get semantic information from semantic image for each point.
3. Point cloud detection
   Semantic point cloud points as input of OpenPCDet model (currently PointPillar) and get 3D detection result. OpenPCDet model need to be trained in advance.


## Requirements

```
Ubuntu System
CUDA 11.1
cudnn 8.0.4
ROS kinetic (for ubuntu16.04) 
python 3.6 (**strictly**, for ROS-python3 environment)
numpy <= 1.20 (this is **crucial**, can use 1.17)
pytorch 1.7
```
ROS has individual version for different ubuntu version. Eg. ros-melodic for ubuntu18.04, ros-noetic for ubuntu20.04.
ROS-python3 environment install guidances below might also work for ros-melodic, because the installation command are summarized from experience of ros-melodic (ubuntu18.04 system support python3.6). For ros-noetic, it support python3 and might not need to configure the environment separately.

For python environment, creatring a conda environment or pip environment of pythonn3.6 is ok.
```
conda create -n <env_name> python=3.6
conda activate <env_name>
```

### Yolov5 Requirements
Refer to requirements.txt in Yolov5 source code.
```
pip install -r requirements.txt
```
Here's some basic modules in need to correctly run this project:
```
pip install argparse pathlib opencv-python pandas requests tqdm matplotlib seaborn
```

### OpenPCDet Requirements
Refer to requirements.txt in OpenPCDet source code.
```
pip install -r requirements.txt
```
And some other modules to install:
```
pip install vtk pyqt5-tools mayavi
```
A sparse convolution module alse in need. For various CUDA can find individual command here.
```
pip install spconv-cu111
```
If **error** "qt.qpa.plugin: Could not load the Qt platform plugin 'xcb' in ..." occurs when running this project, trying to install package below:
```
sudo apt-get install libxcb-xinerama0
```

### ROS Installation
try to better access to website
```
echo "151.101.84.133 raw.github.com" >> /etc/hosts

```
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros-latest.list' 
wget http://packages.ros.org/ros.key -O - | sudo apt-key add - 
sudo apt-get update
sudo apt-get install ros-kinetic-desktop (sudo apt-get install ros-kinetic-desktop-full can completely install but huge)
sudo rosdep init (if slow, see https://www.cnblogs.com/zxzmnh/p/11758103.html)
rosdep update
```
Other ROS packages to install
```
sudo apt-get install ros-kinetic-pcl-conversions
sudo apt-get install ros-kinetic-pcl-ros
sudo apt-get install ros-kinetic-jsk-recognition-msgs
sudo apt-get install ros-kinetic-jsk-rviz-plugins
```

### ROS-python3 environment
Detailed information refers to https://github.com/FunkyCrispy/ROS-Python3-env.git.
To use ROS command eg: rosrun and rostopic, it's known to **"source /opt/ros/kinetic/setup.bash"** first. This will create environment variable **"PYTHONPATH"**. Using "echo $PYTHONPATH" to see its value is **"/opt/ros/kinetic/lib/python2.7/dist-packages"**. Though PYTHONPATH value is null in the begining, python will search site-packages from its paths after sourcing. The order of paths determine which module to be found first if there are several modules with the same name and different version.

When using a compiled ROS project, is known to **"source ./devel/setup.bash" or "source ./install/setup.bash"** first. The relative environment path are added to PYTHONPATH as well. As you can see, the **ROS commands mentioned above can be regard as some pre-defined packages.**

The main idea to solve this ROS-python3 environment is to separately compile ROS source code with python3, and the source code can be regard as packages. When ROS environment of python3 is needed, just source this compiled project is enough, thus original ROS environment of python2 need not to be modified.
Download source code of ROS python, these necessary packages are picked from original source code.
**Recommand to download source code below:**
```
git clone https://github.com/FunkyCrispy/ROS-Python3-env.git
```
or download them all from offical links (but some CMakeLists.txt need to be modified, see https://github.com/FunkyCrispy/ROS-Python3-env.git for detailed information). 
```
git clone https://github.com/ros/ros_comm
git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
git clone https://github.com/ros/geometry
git clone https://github.com/ros/geometry2
```
Compile command (change the python environment path to yours):
**use catkin_make_isolated rather than catkin_make**, in this way tf and tf can contain node with the same name, because the compile separately.
```
catkin_make_isolated -DPYTHON_EXECUTABLE=/usr/local/anaconda3/envs/ros-py36/bin/python -DPYTHON_INCLUDE_DIR=/usr/local/anaconda3/envs/ros-py36/include/python3.6m -DPYTHON_LIBRARY=/usr/local/anaconda3/envs/ros-py36/lib/libpython3.6m.so
```
Then use command below to add project to your environment:
```
source ./devel_isolated/setup.bash
**export PYTHONPATH=/xxx/anaconda/envs/ros-py36/lib/python3.6/site-packages:$PYTHONPATH (otherwise error occurs, probably conflict between ROS-python2.7 and ROS-python3.6)**
```

To use roslaunch in python3, some other lib to install (currently roslaunch is located in compiled project mentioned above):
```
pip install defusedxml netifaces
```
When running roslaunch command, it will add "name" and "log" params after .py scripts, so **if you use argparse lib to pass parameters will cause error**. Recommand to directly use "python sensor_fusion.py" scripts or choose another way to get params, eg: read and get params from a yaml file. 

You can also further directly use ROS command in files to read yaml values into ROS params.

## File Preparation
The weight file path of module yolov5 and OpenPCDet are remained not changed, which may be easier to understand where the weight file is generated. 
**Download or pretrain Yolov5** and place .pt weight file in './src/rospy_sensorfusion/src/yolov5/weights'.

OpenPCDet model's point cloud input dimension are changed from 4 to 8, which means a certain degree of modification are made to project. Point cloud dimensions are changed from (x, y, z, intensity) to (x, y, z, intensity, one-hot), one-hot is for (bg, car, pedestrain, cyclist) 4 classes.

**You can download modified OpenPCDet project though link below. To train OpenPCDet model, you need to download KITTI dataset first and generate dimension extended point cloud files follow this link's guidance.**
```
git clone https://github.com/FunkyCrispy/OpenPCDet-for-fusion.git
```
**Pretrain OpenPCDet and place model .pth weight file** in './src/rospy_sensorfusion/src/OpenPCDet/output/...'(specific path too deep, can be modified) you can define specific path in './src/rospy_sensorfusion/config/OpenPCDet.yaml' as 'ckpt' value.

**Besides, you need to prepare a rosbag .bag file and put it in './src/rospy_sensorfusion/bag', and correspondingly modify './perception.launch' file.**

## Run the project
```
roslaunch perception.launch
```
The environment configuration is tough and tedious, I hope everything goes well and GOOD LUCK!