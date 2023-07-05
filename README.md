# DrapebotTrustAssessment
Software developed in EU Horizon2020 project Drapebot.

The goal of the trust assessment system is to utilize body tracking and potentially other non-intrusive sensors, in combination with the state of the robot and the task context, to calculate the operator’s trust in the robot in real time during the draping process. By tracking the trust level throughout the collaboration session, based on the baseline trust of the operator, we can adapt the robot behavior according to our strategy for maintaining trust. An example of the strategy could be recognizing signs of decreases or disruptions of trust in the robot. When this occurs, we can send a signal to the robot’s operating system that the robot behavior should change, or signal should be sent to the operator via UI to regain trust. 
 ![billede](https://github.com/HRI-AAU/DrapebotTrustAssessment/assets/14195697/95cd86a1-f85c-4380-b47a-1623a785e7fe)

Tracking hardware:
For all sessions, body tracking was performed using the Xsens MVN Awinda tracking suit. It consists of a tight-fitting shirt, gloves, headband, and a series of straps used to attach 17 IMUs to the participant. After calibration the system uses inverse kinematics to track and log the movements of the participant at a rate of 60 measurements per second. The measurements include linear and angular speed, velocity, and acceleration of every skeleton tracking point. For the most accurate tracking the participant’s body dimensions are measured before the session. These measurements include height, foot length, shoulder height, shoulder width, elbow span, wrist span, arm span, hip height, hip width, knee height and ankle height. The system determines position by tracking feet and steps along a plane. This method is, however, vulnerable to position drift over time. The system allows for integration of SteamVR trackers and lighthouses, allowing for position-aiding by attaching a SteamVR-compatible tracker to the participant in addition to the 17 IMUs. 

List of packages in this repository:
custom_msg_hri: contains the ROS message for the XSENS data that is published by the xsensPublisher in MovementTrustAssessment
MovementTrustAssessment: contains the xsensPublisher and xsenseReceiver. xsensReceiver is the node that calculates the features and classifies the data.
(see packages README.md for further instructions)

Additonal information: 
The training data will be available on Zenodo soon (expected: 1 October 2023). Link will be published here.
