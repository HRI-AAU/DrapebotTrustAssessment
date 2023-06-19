# XsensRosPublisher V2.0

Uses the protocol explained in https://www.xsens.com/hubfs/Downloads/Manuals/MVN_real-time_network_streaming_protocol_specification.pdf
Based on our python interface https://github.com/HRI-AAU/Xsens_Python_Interface

Tested on ubuntu20.04    Ros Neotic

# Install

if you haven't already create a catkin workspace
```sh
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
```

clone the repository
```sh
cd src
git clone https://github.com/HRI-AAU/DrapebotTrustAssessment.git
```

build the packages
```sh
cd ..
source devel/setup.bash
catkin_make
```

# Run the ROS side
open a terminal for the ros core
```sh
cd ~/catkin_ws
source devel/setup.bash
roscore
```

open another terminal for the publisher
```sh
cd ~/catkin_ws
source devel/setup.bash
rosrun MovementTrustAssessment xsensePublisher.py
```

open another terminal for the receiver
```sh
cd ~/catkin_ws
source devel/setup.bash
rosrun MovementTrustAssessment xsenseReceiver.py
```

## MVN-Xsens Set up:

In Xsens MVN Analyse go to Options -> Network Streamer

Add a new Stream and set the IP of the client PC. Select the FPS that suits you.

Uncheck all the boxes (datagrams) and add one or many of the following base on your needs:
- Position + Quaternion
- Position + Euler
- Linear Segment Kinematics    ( position , velocity acceleration)
- Angular Segment Kinematics   ( quaternions, ang velocity, ang acceleration)
- Center of Mass

![image](https://user-images.githubusercontent.com/69670188/132003944-16fdeaad-e022-4407-9bd6-b56d6b9c38c5.png)

On the console you can see the active datagrams
![image](https://user-images.githubusercontent.com/69670188/132005615-f9b15b55-3a8b-4aa2-9064-a0228f2d10ad.png)


# Receive the data on your Receiver ROS Node
Use the MovementTrustAssessment/src/xsenseReceiver.py  as a starting point to receive the data

The data from the published topic is defined in src/custom_msg_hri/msg/msg_hri_all_data.msg
```python
time stamp
string frame_id
int32 sample_counter

custom_msg_hri/msg_hri_mass center_of_mass
custom_msg_hri/msg_hri_multi[] joints_data
```

The data for each joint is defined in src/custom_msg_hri/msg/msg_hri_multi.msg
```python
string joint_name

geometry_msgs/Point position

geometry_msgs/Point euler
geometry_msgs/Quaternion quaternion

geometry_msgs/Point velocity
geometry_msgs/Point acceleration

geometry_msgs/Point ang_velocity
geometry_msgs/Point ang_acceleration
```
some examples:
```python
# read center of mass in X
print(data.center_of_mass.center_of_mass.x)
# object with all data of joint 0 ( pelvis id = 1 -1)
pelvis_data = data.joints_data[0]
# re-check we have the pelvis data
print(pelvis_data.joint_name)
# print x,y,z position of pelvis
print(pelvis_data.position.x)
print(pelvis_data.position.y)
print(pelvis_data.position.z)
#print the quaternion values for the pelvis, Qer,Qi,Qj,Qk
print(f"Qer {pelvid_data.quaternion.x}")
print(f"Qi  {pelvid_data.quaternion.y}")
print(f"Qj  {pelvid_data.quaternion.z}")
print(f"Qk  {pelvid_data.quaternion.w}")
``` 
The order in which the joints are appended in the joints_data array is stated in page 13 of the manual.

## Saving CPU and Bandwith if needed
When publishing the data it might be that you don't need some of the data i.e. Angular velocity and acceleration

To save process time and bandwith you can make the message smaller by:

Comment the messages you dont need using # in src/custom_msg_hri/msg/msg_hri_multi.msg
```python
string joint_name

geometry_msgs/Point position

geometry_msgs/Point euler
geometry_msgs/Quaternion quaternion

geometry_msgs/Point velocity
geometry_msgs/Point acceleration

#geometry_msgs/Point ang_velocity
#geometry_msgs/Point ang_acceleration
```
In  MovementTrustAssessment/src/xsensePublisher.py comment the needed lines between 326 and 357 acording to the previous step.
```python
#                    msg_multi.ang_velocity.x = dic_data_lastframe[i+1]["data"]["AVelX"]
#                    msg_multi.ang_velocity.y = dic_data_lastframe[i+1]["data"]["AVelY"]
#                    msg_multi.ang_velocity.z = dic_data_lastframe[i+1]["data"]["AVelZ"]

#                    msg_multi.ang_acceleration.x = dic_data_lastframe[i+1]["data"]["AAccX"]
#                    msg_multi.ang_acceleration.y = dic_data_lastframe[i+1]["data"]["AAccY"]
#                    msg_multi.ang_acceleration.z = dic_data_lastframe[i+1]["data"]["AAccZ"]
                    msg.joints_data.append(msg_multi)
```

rebuild the packages
```sh
cd ~/catkin_ws
source devel/setup.bash
catkin_make
```

### Disclaimer
Quaternions in MVN are saved as Qer,Qi,Qj,Qk they are maped to (x,y,z,w) of the ros msg in that order.

Props read is inplemented but not tested.
This code does not implements any handeling for splited datagrams. This should not be a problem with the configuration above. Adjust your MTU size and/or buffer size in the code if the datagrams are splited.
