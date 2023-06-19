#!/usr/bin/env python3
import rospy
import time
import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
from custom_msg_hri.msg import msg_hri_mass, msg_hri_multi, msg_hri_all_data

#------------------------------------------------------------------------------------------------
#HIGH LEVEL
def vector(x,y,z):  #CHECKED BOTH LOW AND HIGH LEVEL
  V = math.sqrt(pow(x, 2)+pow(y, 2)+pow(z, 2))
  return V
#CoM-Displacement
def CoMD(x1, y1, z1, x2, y2, z2):
  CoMD = vector((x1 - x2), (y1-y2), (z1-z2))
  return CoMD
#Balance
def abs(z1,z2):
  abso =math.sqrt(pow((z1-z2),2))
  return abso
def form(x, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22):
  y1 = abs(x,x1)
  y2 = abs(x,x2)
  y3 = abs(x,x3)
  y4 = abs(x,x4)
  y5 = abs(x,x5)
  y6 = abs(x,x6)
  y7 = abs(x,x7)
  y8 = abs(x,x8)
  y9 = abs(x,x9)
  y10 = abs(x,x10)
  y11 = abs(x,x11)
  y12 = abs(x,x12)
  y13 = abs(x,x13)
  y14 = abs(x,x14)
  y15 = abs(x,x15)
  y16 = abs(x,x16)
  y17 = abs(x,x17)
  y18 = abs(x,x18)
  y19 = abs(x,x19)
  y20 = abs(x,x20)
  y21 = abs(x,x21)
  y22 = abs(x,x22)
  form = max(y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22)
  return form
#distance covered
def distance1(r1,r2,r3,r4):
  dist=math.sqrt(pow((r1-r3),2) + pow((r2-r4),2))
  return dist
#shape directional
def vector1(x,y):
  V = math.sqrt(pow(x, 2)+pow(y, 2))
  return V
def absi(x):
  V = math.sqrt(pow(x, 2))
  return V
def curv(a1,a2,b1,b2): 
  a = a1*b2-a2*b1
  if (b1 !=0) or (b2 !=0):
    curvature = (absi(a))/(pow(vector1(b1,b2), 3))
  else:
    curvature = 0
  return curvature
#LOW LEVEL
#CURVATURE
def curv_2(a1,a2,a3,b1,b2,b3): 
  a = a2*b3-a3*b2
  b = a1*b3-a3*b1
  c = a1*b2-a2*b1
  if ((b1 != 0) or (b2 != 0) or (b3 != 0)):
    curvature_2 = (vector(a,b,c))/(pow(vector(b1,b2,b3), 3))
  else:
    curvature_2 = 0
  return curvature_2
def jerk(a1,a2):
  jerk = (a2-a1)/(1/30)
  return jerk
def sort(x1): 
  lenght=range(len(dfa)) #len(dfa['Frame'])
  l1 = []
  for i in range(len(dfa)):
    if i==0:
      l1.append(jerk(0, x1[i+1]))
    elif i==lenght[-1]:
      l1.append(jerk(x1[i-1], 0))
    else:
      l1.append(jerk(x1[i-1], x1[i+1]))
  return l1
def sort2(x1, y1, z1): 
  l2 = []
  for i in range(len(x1)):
    l2.append(vector(x1[i],y1[i],z1[i]))
  return l2
def dis(x1,y1,z1,x2,y2,z2):
  x = x2-x1
  y = y2-y1
  z = z2-z1
  displacement = vector(x,y,z)
  return displacement
#Effort
#time effort
def sum1(a1,a2,a3,a4):
  s= a1+a2+a3+a4
  return s
def sum2(a1,a2,a3,a4,a5,a6,a7,a8):
  s= a1+a2+a3+a4+a5+a6+a7+a8
  return s
def sum3(x,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22):
  s= x+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19+x20+x21+x22
  return s
#QOM
def qom1(a1,a2,a3,a4):
  s= (a1+2*a2+4*a3+3*a4)/(10)
  return s
def qom2(a1,a2,a3,a4,a5,a6,a7,a8):
  s= (4*a1+3*a2+2*a3+a4+4*a5+3*a6+2*a7+a8)/20
  return s
def qom3(x,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22):
  s= (x+x1*2+x2*4+x3+x4*3+x5+x6*2+x7*4+x8*3+x9*4+x10*3+x11*2+x12+x13*4+x14*3+x15*2+x16*20+x17*5+x18*5+x19*5+x20*5+x21*8+x22*5)/100
  return s

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
model = load_model('**********/best_model.h5') #load model from path
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------


i_g = 0
time0 = 0
def callback(data):
    # Convert the json data into a dictionary
    # The first param is the ID -> index +1 from table in manual page 13
    # https://www.xsens.com/hubfs/Downloads/Manuals/MVN_real-time_network_streaming_protocol_specification.pdf
    global i_g
    global time0
    if(i % 200 == 0):
        com_x=[]
        com_y=[]
        com_z=[]
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        dfa = pd.DataFrame()
        dfV = pd.DataFrame()
        Pelvis_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        L5_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        L3_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        T12_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        T8_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Neck_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Head_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Righ_Shoulder_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Right_Upper_Arm_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Right_Forearm_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Right_Hand_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Left_Shoulder_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Left_Upper_Arm_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Left_Forearm_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Left_Hand_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Right_Upper_Leg_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Right_Lower_Leg_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Right_Foot_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Right_Toe_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Left_Upper_Leg_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Left_Lower_Leg_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Left_Foot_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
        Left_Toe_f = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []} #1=x 2=y 3=z 4,5,6 acc 7,8,9 vel
    i_g+=1
    #print(data.center_of_mass.center_of_mass.x)
    #print(f"Center of mass X = {data.center_of_mass.center_of_mass.y}")
    #print(f"Center of mass X = {data.center_of_mass.center_of_mass.z}")
    #pelvis_data = data.joints_data[5]
    #print(pelvis_data)
    #print(data.joints_data[0])
    com_x.append(data.center_of_mass.center_of_mass.x)
    com_y.append(data.center_of_mass.center_of_mass.y)
    com_z.append(data.center_of_mass.center_of_mass.z)
    Pelvis_f[1].append(data.joints_data[0].position.x)
    Pelvis_f[2].append(data.joints_data[0].position.y)
    Pelvis_f[3].append(data.joints_data[0].position.z)
    Pelvis_f[4].append(data.joints_data[0].acceleration.x)
    Pelvis_f[5].append(data.joints_data[0].acceleration.y)
    Pelvis_f[6].append(data.joints_data[0].acceleration.z)
    Pelvis_f[7].append(data.joints_data[0].velocity.x)
    Pelvis_f[8].append(data.joints_data[0].velocity.y)
    Pelvis_f[9].append(data.joints_data[0].velocity.z)
    L5_f[1].append(data.joints_data[1].position.x)
    L5_f[2].append(data.joints_data[1].position.y)
    L5_f[3].append(data.joints_data[1].position.z)
    L5_f[4].append(data.joints_data[1].acceleration.x)
    L5_f[5].append(data.joints_data[1].acceleration.y)
    L5_f[6].append(data.joints_data[1].acceleration.z)
    L5_f[7].append(data.joints_data[1].velocity.x)
    L5_f[8].append(data.joints_data[1].velocity.y)
    L5_f[9].append(data.joints_data[1].velocity.z)
    L3_f[1].append(data.joints_data[2].position.x)
    L3_f[2].append(data.joints_data[2].position.y)
    L3_f[3].append(data.joints_data[2].position.z)
    L3_f[4].append(data.joints_data[2].acceleration.x)
    L3_f[5].append(data.joints_data[2].acceleration.y)
    L3_f[6].append(data.joints_data[2].acceleration.z)
    L3_f[7].append(data.joints_data[2].velocity.x)
    L3_f[8].append(data.joints_data[2].velocity.y)
    L3_f[9].append(data.joints_data[2].velocity.z)
    T12_f[1].append(data.joints_data[3].position.x)
    T12_f[2].append(data.joints_data[3].position.y)
    T12_f[3].append(data.joints_data[3].position.z)
    T12_f[4].append(data.joints_data[3].acceleration.x)
    T12_f[5].append(data.joints_data[3].acceleration.y)
    T12_f[6].append(data.joints_data[3].acceleration.z)
    T12_f[7].append(data.joints_data[3].velocity.x)
    T12_f[8].append(data.joints_data[3].velocity.y)
    T12_f[9].append(data.joints_data[3].velocity.z)
    T8_f[1].append(data.joints_data[4].position.x)
    T8_f[2].append(data.joints_data[4].position.y)
    T8_f[3].append(data.joints_data[4].position.z)
    T8_f[4].append(data.joints_data[4].acceleration.x)
    T8_f[5].append(data.joints_data[4].acceleration.y)
    T8_f[6].append(data.joints_data[4].acceleration.z)
    T8_f[7].append(data.joints_data[4].velocity.x)
    T8_f[8].append(data.joints_data[4].velocity.y)
    T8_f[9].append(data.joints_data[4].velocity.z)
    Neck_f[1].append(data.joints_data[5].position.x)
    Neck_f[2].append(data.joints_data[5].position.y)
    Neck_f[3].append(data.joints_data[5].position.z)
    Neck_f[4].append(data.joints_data[5].acceleration.x)
    Neck_f[5].append(data.joints_data[5].acceleration.y)
    Neck_f[6].append(data.joints_data[5].acceleration.z)
    Neck_f[7].append(data.joints_data[5].velocity.x)
    Neck_f[8].append(data.joints_data[5].velocity.y)
    Neck_f[9].append(data.joints_data[5].velocity.z)
    Head_f[1].append(data.joints_data[6].position.x)
    Head_f[2].append(data.joints_data[6].position.y)
    Head_f[3].append(data.joints_data[6].position.z)
    Head_f[4].append(data.joints_data[6].acceleration.x)
    Head_f[5].append(data.joints_data[6].acceleration.y)
    Head_f[6].append(data.joints_data[6].acceleration.z)
    Head_f[7].append(data.joints_data[6].velocity.x)
    Head_f[8].append(data.joints_data[6].velocity.y)
    Head_f[9].append(data.joints_data[6].velocity.z)
    Righ_Shoulder_f[1].append(data.joints_data[7].position.x)
    Righ_Shoulder_f[2].append(data.joints_data[7].position.y)
    Righ_Shoulder_f[3].append(data.joints_data[7].position.z)
    Righ_Shoulder_f[4].append(data.joints_data[7].acceleration.x)
    Righ_Shoulder_f[5].append(data.joints_data[7].acceleration.y)
    Righ_Shoulder_f[6].append(data.joints_data[7].acceleration.z)
    Righ_Shoulder_f[7].append(data.joints_data[7].velocity.x)
    Righ_Shoulder_f[8].append(data.joints_data[7].velocity.y)
    Righ_Shoulder_f[9].append(data.joints_data[7].velocity.z)
    Right_Upper_Arm_f[1].append(data.joints_data[8].position.x)
    Right_Upper_Arm_f[2].append(data.joints_data[8].position.y)
    Right_Upper_Arm_f[3].append(data.joints_data[8].position.z)
    Right_Upper_Arm_f[4].append(data.joints_data[8].acceleration.x)
    Right_Upper_Arm_f[5].append(data.joints_data[8].acceleration.y)
    Right_Upper_Arm_f[6].append(data.joints_data[8].acceleration.z)
    Right_Upper_Arm_f[7].append(data.joints_data[8].velocity.x)
    Right_Upper_Arm_f[8].append(data.joints_data[8].velocity.y)
    Right_Upper_Arm_f[9].append(data.joints_data[8].velocity.z)
    Right_Forearm_f[1].append(data.joints_data[9].position.x)
    Right_Forearm_f[2].append(data.joints_data[9].position.y)
    Right_Forearm_f[3].append(data.joints_data[9].position.z)
    Right_Forearm_f[4].append(data.joints_data[9].acceleration.x)
    Right_Forearm_f[5].append(data.joints_data[9].acceleration.y)
    Right_Forearm_f[6].append(data.joints_data[9].acceleration.z)
    Right_Forearm_f[7].append(data.joints_data[9].velocity.x)
    Right_Forearm_f[8].append(data.joints_data[9].velocity.y)
    Right_Forearm_f[9].append(data.joints_data[9].velocity.z)
    Right_Hand_f[1].append(data.joints_data[10].position.x)
    Right_Hand_f[2].append(data.joints_data[10].position.y)
    Right_Hand_f[3].append(data.joints_data[10].position.z)
    Right_Hand_f[4].append(data.joints_data[10].acceleration.x)
    Right_Hand_f[5].append(data.joints_data[10].acceleration.y)
    Right_Hand_f[6].append(data.joints_data[10].acceleration.z)
    Right_Hand_f[7].append(data.joints_data[10].velocity.x)
    Right_Hand_f[8].append(data.joints_data[10].velocity.y)
    Right_Hand_f[9].append(data.joints_data[10].velocity.z)
    Left_Shoulder_f[1].append(data.joints_data[11].position.x)
    Left_Shoulder_f[2].append(data.joints_data[11].position.y)
    Left_Shoulder_f[3].append(data.joints_data[11].position.z)
    Left_Shoulder_f[4].append(data.joints_data[11].acceleration.x)
    Left_Shoulder_f[5].append(data.joints_data[11].acceleration.y)
    Left_Shoulder_f[6].append(data.joints_data[11].acceleration.z)
    Left_Shoulder_f[7].append(data.joints_data[11].velocity.x)
    Left_Shoulder_f[8].append(data.joints_data[11].velocity.y)
    Left_Shoulder_f[9].append(data.joints_data[11].velocity.z)
    Left_Upper_Arm_f[1].append(data.joints_data[12].position.x)
    Left_Upper_Arm_f[2].append(data.joints_data[12].position.y)
    Left_Upper_Arm_f[3].append(data.joints_data[12].position.z)
    Left_Upper_Arm_f[4].append(data.joints_data[12].acceleration.x)
    Left_Upper_Arm_f[5].append(data.joints_data[12].acceleration.y)
    Left_Upper_Arm_f[6].append(data.joints_data[12].acceleration.z)
    Left_Upper_Arm_f[7].append(data.joints_data[12].velocity.x)
    Left_Upper_Arm_f[8].append(data.joints_data[12].velocity.y)
    Left_Upper_Arm_f[9].append(data.joints_data[12].velocity.z)
    Left_Forearm_f[1].append(data.joints_data[13].position.x)
    Left_Forearm_f[2].append(data.joints_data[13].position.y)
    Left_Forearm_f[3].append(data.joints_data[13].position.z)
    Left_Forearm_f[4].append(data.joints_data[13].acceleration.x)
    Left_Forearm_f[5].append(data.joints_data[13].acceleration.y)
    Left_Forearm_f[6].append(data.joints_data[13].acceleration.z)
    Left_Forearm_f[7].append(data.joints_data[13].velocity.x)
    Left_Forearm_f[8].append(data.joints_data[13].velocity.y)
    Left_Forearm_f[9].append(data.joints_data[13].velocity.z)
    Left_Hand_f[1].append(data.joints_data[14].position.x)
    Left_Hand_f[2].append(data.joints_data[14].position.y)
    Left_Hand_f[3].append(data.joints_data[14].position.z)
    Left_Hand_f[4].append(data.joints_data[14].acceleration.x)
    Left_Hand_f[5].append(data.joints_data[14].acceleration.y)
    Left_Hand_f[6].append(data.joints_data[14].acceleration.z)
    Left_Hand_f[7].append(data.joints_data[14].velocity.x)
    Left_Hand_f[8].append(data.joints_data[14].velocity.y)
    Left_Hand_f[9].append(data.joints_data[14].velocity.z)
    Right_Upper_Leg_f[1].append(data.joints_data[15].position.x)
    Right_Upper_Leg_f[2].append(data.joints_data[15].position.y)
    Right_Upper_Leg_f[3].append(data.joints_data[15].position.z)
    Right_Upper_Leg_f[4].append(data.joints_data[15].acceleration.x)
    Right_Upper_Leg_f[5].append(data.joints_data[15].acceleration.y)
    Right_Upper_Leg_f[6].append(data.joints_data[15].acceleration.z)
    Right_Upper_Leg_f[7].append(data.joints_data[15].velocity.x)
    Right_Upper_Leg_f[8].append(data.joints_data[15].velocity.y)
    Right_Upper_Leg_f[9].append(data.joints_data[15].velocity.z)
    Right_Lower_Leg_f[1].append(data.joints_data[16].position.x)
    Right_Lower_Leg_f[2].append(data.joints_data[16].position.y)
    Right_Lower_Leg_f[3].append(data.joints_data[16].position.z)
    Right_Lower_Leg_f[4].append(data.joints_data[16].acceleration.x)
    Right_Lower_Leg_f[5].append(data.joints_data[16].acceleration.y)
    Right_Lower_Leg_f[6].append(data.joints_data[16].acceleration.z)
    Right_Lower_Leg_f[7].append(data.joints_data[16].velocity.x)
    Right_Lower_Leg_f[8].append(data.joints_data[16].velocity.y)
    Right_Lower_Leg_f[9].append(data.joints_data[16].velocity.z)
    Right_Foot_f[1].append(data.joints_data[17].position.x)
    Right_Foot_f[2].append(data.joints_data[17].position.y)
    Right_Foot_f[3].append(data.joints_data[17].position.z)
    Right_Foot_f[4].append(data.joints_data[17].acceleration.x)
    Right_Foot_f[5].append(data.joints_data[17].acceleration.y)
    Right_Foot_f[6].append(data.joints_data[17].acceleration.z)
    Right_Foot_f[7].append(data.joints_data[17].velocity.x)
    Right_Foot_f[8].append(data.joints_data[17].velocity.y)
    Right_Foot_f[9].append(data.joints_data[17].velocity.z)
    Right_Toe_f[1].append(data.joints_data[18].position.x)
    Right_Toe_f[2].append(data.joints_data[18].position.y)
    Right_Toe_f[3].append(data.joints_data[18].position.z)
    Right_Toe_f[4].append(data.joints_data[18].acceleration.x)
    Right_Toe_f[5].append(data.joints_data[18].acceleration.y)
    Right_Toe_f[6].append(data.joints_data[18].acceleration.z)
    Right_Toe_f[7].append(data.joints_data[18].velocity.x)
    Right_Toe_f[8].append(data.joints_data[18].velocity.y)
    Right_Toe_f[9].append(data.joints_data[18].velocity.z)
    Left_Upper_Leg_f[1].append(data.joints_data[19].position.x)
    Left_Upper_Leg_f[2].append(data.joints_data[19].position.y)
    Left_Upper_Leg_f[3].append(data.joints_data[19].position.z)
    Left_Upper_Leg_f[4].append(data.joints_data[19].acceleration.x)
    Left_Upper_Leg_f[5].append(data.joints_data[19].acceleration.y)
    Left_Upper_Leg_f[6].append(data.joints_data[19].acceleration.z)
    Left_Upper_Leg_f[7].append(data.joints_data[19].velocity.x)
    Left_Upper_Leg_f[8].append(data.joints_data[19].velocity.y)
    Left_Upper_Leg_f[9].append(data.joints_data[19].velocity.z)
    Left_Lower_Leg_f[1].append(data.joints_data[20].position.x)
    Left_Lower_Leg_f[2].append(data.joints_data[20].position.y)
    Left_Lower_Leg_f[3].append(data.joints_data[20].position.z)
    Left_Lower_Leg_f[4].append(data.joints_data[20].acceleration.x)
    Left_Lower_Leg_f[5].append(data.joints_data[20].acceleration.y)
    Left_Lower_Leg_f[6].append(data.joints_data[20].acceleration.z)
    Left_Lower_Leg_f[7].append(data.joints_data[20].velocity.x)
    Left_Lower_Leg_f[8].append(data.joints_data[20].velocity.y)
    Left_Lower_Leg_f[9].append(data.joints_data[20].velocity.z)
    Left_Foot_f[1].append(data.joints_data[21].position.x)
    Left_Foot_f[2].append(data.joints_data[21].position.y)
    Left_Foot_f[3].append(data.joints_data[21].position.z)
    Left_Foot_f[4].append(data.joints_data[21].acceleration.x)
    Left_Foot_f[5].append(data.joints_data[21].acceleration.y)
    Left_Foot_f[6].append(data.joints_data[21].acceleration.z)
    Left_Foot_f[7].append(data.joints_data[21].velocity.x)
    Left_Foot_f[8].append(data.joints_data[21].velocity.y)
    Left_Foot_f[9].append(data.joints_data[21].velocity.z)
    Left_Toe_f[1].append(data.joints_data[22].position.x)
    Left_Toe_f[2].append(data.joints_data[22].position.y)
    Left_Toe_f[3].append(data.joints_data[22].position.z)
    Left_Toe_f[4].append(data.joints_data[22].acceleration.x)
    Left_Toe_f[5].append(data.joints_data[22].acceleration.y)
    Left_Toe_f[6].append(data.joints_data[22].acceleration.z)
    Left_Toe_f[7].append(data.joints_data[22].velocity.x)
    Left_Toe_f[8].append(data.joints_data[22].velocity.y)
    Left_Toe_f[9].append(data.joints_data[22].velocity.z)

    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #TO BE CONTINUED
    
    data.center_of_mass.center_of_mass.y
    if(i_g % 200 == 0):
        df2['CoM pos x'] = com_x
        df2['CoM pos y'] = com_y
        df2['CoM pos z'] = com_z
        df['Pelvis x'] = Pelvis_f[1]
        df['Pelvis y'] = Pelvis_f[2]
        df['Pelvis z'] = Pelvis_f[3]
        df['L5 x'] = L5_f[1]
        df['L5 y'] = L5_f[2]
        df['L5 z'] = L5_f[3]
        df['L3 x'] = L3_f[1]
        df['L3 y'] = L3_f[2]
        df['L3 z'] = L3_f[3]
        df['T12 x'] = T12_f[1]
        df['T12 y'] = T12_f[2]
        df['T12 z'] = T12_f[3]
        df['T8 x'] = T8_f[1]
        df['T8 y'] = T8_f[2]
        df['T8 z'] = T8_f[3]
        df['Neck x'] = Neck_f[1]
        df['Neck y'] = Neck_f[2]
        df['Neck z'] = Neck_f[3]
        df['Head x'] = Head_f[1]
        df['Head y'] = Head_f[2]
        df['Head z'] = Head_f[3]
        df['Right Shoulder x'] = Righ_Shoulder_f[1]
        df['Right Shoulder y'] = Righ_Shoulder_f[2]
        df['Right Shoulder z'] = Righ_Shoulder_f[3]
        df['Right Upper Arm x'] = Right_Upper_Arm_f[1]
        df['Right Upper Arm y'] = Right_Upper_Arm_f[2]
        df['Right Upper Arm z'] = Right_Upper_Arm_f[3]
        df['Right Forearm x'] = Right_Forearm_f[1]
        df['Right Forearm y'] = Right_Forearm_f[2]
        df['Right Forearm z'] = Right_Forearm_f[3]
        df['Right Hand x'] = Right_Hand_f[1]
        df['Right Hand y'] = Right_Hand_f[2]
        df['Right Hand z'] = Right_Hand_f[3]
        df['Left Shoulder x'] = Left_Shoulder_f[1]
        df['Left Shoulder y'] = Left_Shoulder_f[2]
        df['Left Shoulder z'] = Left_Shoulder_f[3]
        df['Left Upper Arm x'] = Left_Upper_Arm_f[1]
        df['Left Upper Arm y'] = Left_Upper_Arm_f[2]
        df['Left Upper Arm z'] = Left_Upper_Arm_f[3]
        df['Left Forearm x'] = Left_Forearm_f[1]
        df['Left Forearm y'] = Left_Forearm_f[2]
        df['Left Forearm z'] = Left_Forearm_f[3]
        df['Left Hand x'] = Left_Hand_f[1]
        df['Left Hand y'] = Left_Hand_f[2]
        df['Left Hand z'] = Left_Hand_f[3]
        df['Right Upper Leg x'] = Right_Upper_Leg_f[1]
        df['Right Upper Leg y'] = Right_Upper_Leg_f[2]
        df['Right Upper Leg z'] = Right_Upper_Leg_f[3]
        df['Right Lower Leg x'] = Right_Lower_Leg_f[1]
        df['Right Lower Leg y'] = Right_Lower_Leg_f[2]
        df['Right Lower Leg z'] = Right_Lower_Leg_f[3]
        df['Right Foot x'] = Right_Foot_f[1]
        df['Right Foot y'] = Right_Foot_f[2]
        df['Right Foot z'] = Right_Foot_f[3]
        df['Right Toe x'] = Right_Toe_f[1]
        df['Right Toe y'] = Right_Toe_f[2]
        df['Right Toe z'] = Right_Toe_f[3]
        df['Left Upper Leg x'] = Left_Upper_Leg_f[1]
        df['Left Upper Leg y'] = Left_Upper_Leg_f[2]
        df['Left Upper Leg z'] = Left_Upper_Leg_f[3]
        df['Left Lower Leg x'] = Left_Lower_Leg_f[1]
        df['Left Lower Leg y'] = Left_Lower_Leg_f[2]
        df['Left Lower Leg z'] = Left_Lower_Leg_f[3]
        df['Left Foot x'] = Left_Foot_f[1]
        df['Left Foot y'] = Left_Foot_f[2]
        df['Left Foot z'] = Left_Foot_f[3]
        df['Left Toe x'] = Left_Toe_f[1]
        df['Left Toe y'] = Left_Toe_f[2]
        df['Left Toe z'] = Left_Toe_f[3]

        dfa['Pelvis x'] = Pelvis_f[4]
        dfa['Pelvis y'] = Pelvis_f[5]
        dfa['Pelvis z'] = Pelvis_f[6]
        dfa['L5 x'] = L5_f[4]
        dfa['L5 y'] = L5_f[5]
        dfa['L5 z'] = L5_f[6]
        dfa['L3 x'] = L3_f[4]
        dfa['L3 y'] = L3_f[5]
        dfa['L3 z'] = L3_f[6]
        dfa['T12 x'] = T12_f[4]
        dfa['T12 y'] = T12_f[5]
        dfa['T12 z'] = T12_f[6]
        dfa['T8 x'] = T8_f[4]
        dfa['T8 y'] = T8_f[5]
        dfa['T8 z'] = T8_f[6]
        dfa['Neck x'] = Neck_f[4]
        dfa['Neck y'] = Neck_f[5]
        dfa['Neck z'] = Neck_f[6]
        dfa['Head x'] = Head_f[4]
        dfa['Head y'] = Head_f[5]
        dfa['Head z'] = Head_f[6]
        dfa['Right Shoulder x'] = Righ_Shoulder_f[4]
        dfa['Right Shoulder y'] = Righ_Shoulder_f[5]
        dfa['Right Shoulder z'] = Righ_Shoulder_f[6]
        dfa['Right Upper Arm x'] = Right_Upper_Arm_f[4]
        dfa['Right Upper Arm y'] = Right_Upper_Arm_f[5]
        dfa['Right Upper Arm z'] = Right_Upper_Arm_f[6]
        dfa['Right Forearm x'] = Right_Forearm_f[4]
        dfa['Right Forearm y'] = Right_Forearm_f[5]
        dfa['Right Forearm z'] = Right_Forearm_f[6]
        dfa['Right Hand x'] = Right_Hand_f[4]
        dfa['Right Hand y'] = Right_Hand_f[5]
        dfa['Right Hand z'] = Right_Hand_f[6]
        dfa['Left Shoulder x'] = Left_Shoulder_f[4]
        dfa['Left Shoulder y'] = Left_Shoulder_f[5]
        dfa['Left Shoulder z'] = Left_Shoulder_f[6]
        dfa['Left Upper Arm x'] = Left_Upper_Arm_f[4]
        dfa['Left Upper Arm y'] = Left_Upper_Arm_f[5]
        dfa['Left Upper Arm z'] = Left_Upper_Arm_f[6]
        dfa['Left Forearm x'] = Left_Forearm_f[4]
        dfa['Left Forearm y'] = Left_Forearm_f[5]
        dfa['Left Forearm z'] = Left_Forearm_f[6]
        dfa['Left Hand x'] = Left_Hand_f[4]
        dfa['Left Hand y'] = Left_Hand_f[5]
        dfa['Left Hand z'] = Left_Hand_f[6]
        dfa['Right Upper Leg x'] = Right_Upper_Leg_f[4]
        dfa['Right Upper Leg y'] = Right_Upper_Leg_f[5]
        dfa['Right Upper Leg z'] = Right_Upper_Leg_f[6]
        dfa['Right Lower Leg x'] = Right_Lower_Leg_f[4]
        dfa['Right Lower Leg y'] = Right_Lower_Leg_f[5]
        dfa['Right Lower Leg z'] = Right_Lower_Leg_f[6]
        dfa['Right Foot x'] = Right_Foot_f[4]
        dfa['Right Foot y'] = Right_Foot_f[5]
        dfa['Right Foot z'] = Right_Foot_f[6]
        dfa['Right Toe x'] = Right_Toe_f[4]
        dfa['Right Toe y'] = Right_Toe_f[5]
        dfa['Right Toe z'] = Right_Toe_f[6]
        dfa['Left Upper Leg x'] = Left_Upper_Leg_f[4]
        dfa['Left Upper Leg y'] = Left_Upper_Leg_f[5]
        dfa['Left Upper Leg z'] = Left_Upper_Leg_f[6]
        dfa['Left Lower Leg x'] = Left_Lower_Leg_f[4]
        dfa['Left Lower Leg y'] = Left_Lower_Leg_f[5]
        dfa['Left Lower Leg z'] = Left_Lower_Leg_f[6]
        dfa['Left Foot x'] = Left_Foot_f[4]
        dfa['Left Foot y'] = Left_Foot_f[5]
        dfa['Left Foot z'] = Left_Foot_f[6]
        dfa['Left Toe x'] = Left_Toe_f[4]
        dfa['Left Toe y'] = Left_Toe_f[5]
        dfa['Left Toe z'] = Left_Toe_f[6]
        dfV['Pelvis x'] = Pelvis_f[7]
        dfV['Pelvis y'] = Pelvis_f[8]
        dfV['Pelvis z'] = Pelvis_f[9]
        dfV['L5 x'] = L5_f[7]
        dfV['L5 y'] = L5_f[8]
        dfV['L5 z'] = L5_f[9]
        dfV['L3 x'] = L3_f[7]
        dfV['L3 y'] = L3_f[8]
        dfV['L3 z'] = L3_f[9]
        dfV['T12 x'] = T12_f[7]
        dfV['T12 y'] = T12_f[8]
        dfV['T12 z'] = T12_f[9]
        dfV['T8 x'] = T8_f[7]
        dfV['T8 y'] = T8_f[8]
        dfV['T8 z'] = T8_f[9]
        dfV['Neck x'] = Neck_f[7]
        dfV['Neck y'] = Neck_f[8]
        dfV['Neck z'] = Neck_f[9]
        dfV['Head x'] = Head_f[7]
        dfV['Head y'] = Head_f[8]
        dfV['Head z'] = Head_f[9]
        dfV['Right Shoulder x'] = Righ_Shoulder_f[7]
        dfV['Right Shoulder y'] = Righ_Shoulder_f[8]
        dfV['Right Shoulder z'] = Righ_Shoulder_f[9]
        dfV['Right Upper Arm x'] = Right_Upper_Arm_f[7]
        dfV['Right Upper Arm y'] = Right_Upper_Arm_f[8]
        dfV['Right Upper Arm z'] = Right_Upper_Arm_f[9]
        dfV['Right Forearm x'] = Right_Forearm_f[7]
        dfV['Right Forearm y'] = Right_Forearm_f[8]
        dfV['Right Forearm z'] = Right_Forearm_f[9]
        dfV['Right Hand x'] = Right_Hand_f[7]
        dfV['Right Hand y'] = Right_Hand_f[8]
        dfV['Right Hand z'] = Right_Hand_f[9]
        dfV['Left Shoulder x'] = Left_Shoulder_f[7]
        dfV['Left Shoulder y'] = Left_Shoulder_f[8]
        dfV['Left Shoulder z'] = Left_Shoulder_f[9]
        dfV['Left Upper Arm x'] = Left_Upper_Arm_f[7]
        dfV['Left Upper Arm y'] = Left_Upper_Arm_f[8]
        dfV['Left Upper Arm z'] = Left_Upper_Arm_f[9]
        dfV['Left Forearm x'] = Left_Forearm_f[7]
        dfV['Left Forearm y'] = Left_Forearm_f[8]
        dfV['Left Forearm z'] = Left_Forearm_f[9]
        dfV['Left Hand x'] = Left_Hand_f[7]
        dfV['Left Hand y'] = Left_Hand_f[8]
        dfV['Left Hand z'] = Left_Hand_f[9]
        dfV['Right Upper Leg x'] = Right_Upper_Leg_f[7]
        dfV['Right Upper Leg y'] = Right_Upper_Leg_f[8]
        dfV['Right Upper Leg z'] = Right_Upper_Leg_f[9]
        dfV['Right Lower Leg x'] = Right_Lower_Leg_f[7]
        dfV['Right Lower Leg y'] = Right_Lower_Leg_f[8]
        dfV['Right Lower Leg z'] = Right_Lower_Leg_f[9]
        dfV['Right Foot x'] = Right_Foot_f[7]
        dfV['Right Foot y'] = Right_Foot_f[8]
        dfV['Right Foot z'] = Right_Foot_f[9]
        dfV['Right Toe x'] = Right_Toe_f[7]
        dfV['Right Toe y'] = Right_Toe_f[8]
        dfV['Right Toe z'] = Right_Toe_f[9]
        dfV['Left Upper Leg x'] = Left_Upper_Leg_f[7]
        dfV['Left Upper Leg y'] = Left_Upper_Leg_f[8]
        dfV['Left Upper Leg z'] = Left_Upper_Leg_f[9]
        dfV['Left Lower Leg x'] = Left_Lower_Leg_f[7]
        dfV['Left Lower Leg y'] = Left_Lower_Leg_f[8]
        dfV['Left Lower Leg z'] = Left_Lower_Leg_f[9]
        dfV['Left Foot x'] = Left_Foot_f[7]
        dfV['Left Foot y'] = Left_Foot_f[8]
        dfV['Left Foot z'] = Left_Foot_f[9]
        dfV['Left Toe x'] = Left_Toe_f[7]
        dfV['Left Toe y'] = Left_Toe_f[8]
        dfV['Left Toe z'] = Left_Toe_f[9]
        
    

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

        #print(f"Time since last 60 packages{time0 - time.time()}")
        #print(f"received = {i} pcgs  Xsens sample = {data.sample_counter}")
        #print(f"Data from {data.joints_data[0].joint_name}")
        #print()
        #print(f"Pelvis Quaternion Qer= {data.joints_data[0].quaternion.x}")
        #print(f"Center of mass X = {data.center_of_mass.center_of_mass.x}")
        
        #LOW LEVEL
        #CURVATURE
        PelvisC = []
        for i in range(len(dfa)):
          PelvisC.append(curv_2(dfa['Pelvis x'][i], dfa['Pelvis y'][i], dfa['Pelvis z'][i], dfV['Pelvis x'][i], dfV['Pelvis y'][i], dfV['Pelvis z'][i]))
        L5C = []
        for i in range(len(dfa)):
          L5C.append(curv_2(dfa['L5 x'][i], dfa['L5 y'][i], dfa['L5 z'][i], dfV['L5 x'][i], dfV['L5 y'][i], dfV['L5 z'][i]))
        L3C = []
        for i in range(len(dfa)):
          L3C.append(curv_2(dfa['L3 x'][i], dfa['L3 y'][i], dfa['L3 z'][i], dfV['L3 x'][i], dfV['L3 y'][i], dfV['L3 z'][i]))
        T12C = []
        for i in range(len(dfa)):
          T12C.append(curv_2(dfa['T12 x'][i], dfa['T12 y'][i], dfa['T12 z'][i], dfV['T12 x'][i], dfV['T12 y'][i], dfV['T12 z'][i]))
        T8C = []
        for i in range(len(dfa)):
          T8C.append(curv_2(dfa['T8 x'][i], dfa['T8 y'][i], dfa['T8 z'][i], dfV['T8 x'][i], dfV['T8 y'][i], dfV['T8 z'][i]))
        NeckC = []
        for i in range(len(dfa)):
          NeckC.append(curv_2(dfa['Neck x'][i], dfa['Neck y'][i], dfa['Neck z'][i], dfV['Neck x'][i], dfV['Neck y'][i], dfV['Neck z'][i]))
        HeadC = []
        for i in range(len(dfa)):
          HeadC.append(curv_2(dfa['Head x'][i], dfa['Head y'][i], dfa['Head z'][i], dfV['Head x'][i], dfV['Head y'][i], dfV['Head z'][i]))
        RightShoulderC = []
        for i in range(len(dfa)):
            RightShoulderC.append(curv_2(dfa['Right Shoulder x'][i], dfa['Right Shoulder y'][i], dfa['Right Shoulder z'][i], dfV['Right Shoulder x'][i], dfV['Right Shoulder y'][i], dfV['Right Shoulder z'][i]))
        RightUpperArmC = []
        for i in range(len(dfa)):
            RightUpperArmC.append(curv_2(dfa['Right Upper Arm x'][i], dfa['Right Upper Arm y'][i], dfa['Right Upper Arm z'][i], dfV['Right Upper Arm x'][i], dfV['Right Upper Arm y'][i], dfV['Right Upper Arm z'][i]))
        RightForearmC = []
        for i in range(len(dfa)):
            RightForearmC.append(curv_2(dfa['Right Forearm x'][i], dfa['Right Forearm y'][i], dfa['Right Forearm z'][i], dfV['Right Forearm x'][i], dfV['Right Forearm y'][i], dfV['Right Forearm z'][i]))
        RightHandC = []
        for i in range(len(dfa)):
            RightHandC.append(curv_2(dfa['Right Hand x'][i], dfa['Right Hand y'][i], dfa['Right Hand z'][i], dfV['Right Hand x'][i], dfV['Right Hand y'][i], dfV['Right Hand z'][i]))
        LeftShoulderC = []
        for i in range(len(dfa)):
            LeftShoulderC.append(curv_2(dfa['Left Shoulder x'][i], dfa['Left Shoulder y'][i], dfa['Left Shoulder z'][i], dfV['Left Shoulder x'][i], dfV['Left Shoulder y'][i], dfV['Left Shoulder z'][i]))
        LeftUpperArmC = []
        for i in range(len(dfa)):
            LeftUpperArmC.append(curv_2(dfa['Left Upper Arm x'][i], dfa['Left Upper Arm y'][i], dfa['Left Upper Arm z'][i], dfV['Left Upper Arm x'][i], dfV['Left Upper Arm y'][i], dfV['Left Upper Arm z'][i]))
        LeftForearmC = []
        for i in range(len(dfa)):
            LeftForearmC.append(curv_2(dfa['Left Forearm x'][i], dfa['Left Forearm y'][i], dfa['Left Forearm z'][i], dfV['Left Forearm x'][i], dfV['Left Forearm y'][i], dfV['Left Forearm z'][i]))
        LeftHandC = []
        for i in range(len(dfa)):
            LeftHandC.append(curv_2(dfa['Left Hand x'][i], dfa['Left Hand y'][i], dfa['Left Hand z'][i], dfV['Left Hand x'][i], dfV['Left Hand y'][i], dfV['Left Hand z'][i]))
        RightUpperLegC = []
        for i in range(len(dfa)):
            RightUpperLegC.append(curv_2(dfa['Right Upper Leg x'][i], dfa['Right Upper Leg y'][i], dfa['Right Upper Leg z'][i], dfV['Right Upper Leg x'][i], dfV['Right Upper Leg y'][i], dfV['Right Upper Leg z'][i]))
        RightLowerLegC = []
        for i in range(len(dfa)):
            RightLowerLegC.append(curv_2(dfa['Right Lower Leg x'][i], dfa['Right Lower Leg y'][i], dfa['Right Lower Leg z'][i], dfV['Right Lower Leg x'][i], dfV['Right Lower Leg y'][i], dfV['Right Lower Leg z'][i]))
        RightFootC = []
        for i in range(len(dfa)):
            RightFootC.append(curv_2(dfa['Right Foot x'][i], dfa['Right Foot y'][i], dfa['Right Foot z'][i], dfV['Right Foot x'][i], dfV['Right Foot y'][i], dfV['Right Foot z'][i]))
        RightToeC = []
        for i in range(len(dfa)):
            RightToeC.append(curv_2(dfa['Right Toe x'][i], dfa['Right Toe y'][i], dfa['Right Toe z'][i], dfV['Right Toe x'][i], dfV['Right Toe y'][i], dfV['Right Toe z'][i]))
        LeftUpperLegC = []
        for i in range(len(dfa)):
            LeftUpperLegC.append(curv_2(dfa['Left Upper Leg x'][i], dfa['Left Upper Leg y'][i], dfa['Left Upper Leg z'][i], dfV['Left Upper Leg x'][i], dfV['Left Upper Leg y'][i], dfV['Left Upper Leg z'][i]))
        LeftLowerLegC = []
        for i in range(len(dfa)):
            LeftLowerLegC.append(curv_2(dfa['Left Lower Leg x'][i], dfa['Left Lower Leg y'][i], dfa['Left Lower Leg z'][i], dfV['Left Lower Leg x'][i], dfV['Left Lower Leg y'][i], dfV['Left Lower Leg z'][i]))
        LeftFootC = []
        for i in range(len(dfa)):
            LeftFootC.append(curv_2(dfa['Left Foot x'][i], dfa['Left Foot y'][i], dfa['Left Foot z'][i], dfV['Left Foot x'][i], dfV['Left Foot y'][i], dfV['Left Foot z'][i]))
        LeftToeC = []
        for i in range(len(dfa)):
            LeftToeC.append(curv_2(dfa['Left Toe x'][i], dfa['Left Toe y'][i], dfa['Left Toe z'][i], dfV['Left Toe x'][i], dfV['Left Toe y'][i], dfV['Left Toe z'][i]))
        #---------------------------------------------------------------------------------------------
        # ACC AND VEL
        PelvisA = []
        for i in range(len(dfa)):
            PelvisA.append(vector(dfa['Pelvis x'][i], dfa['Pelvis y'][i], dfa['Pelvis z'][i]))
        L5A = []
        for i in range(len(dfa)):
            L5A.append(vector(dfa['L5 x'][i], dfa['L5 y'][i], dfa['L5 z'][i]))
        L3A = []
        for i in range(len(dfa)):
            L3A.append(vector(dfa['L3 x'][i], dfa['L3 y'][i], dfa['L3 z'][i]))
        T12A = []
        for i in range(len(dfa)):
            T12A.append(vector(dfa['T12 x'][i], dfa['T12 y'][i], dfa['T12 z'][i]))
        T8A = []
        for i in range(len(dfa)):
            T8A.append(vector(dfa['T8 x'][i], dfa['T8 y'][i], dfa['T8 z'][i]))
        NeckA = []
        for i in range(len(dfa)):
            NeckA.append(vector(dfa['Neck x'][i], dfa['Neck y'][i], dfa['Neck z'][i]))
        HeadA = []
        for i in range(len(dfa)):
            HeadA.append(vector(dfa['Head x'][i], dfa['Head y'][i], dfa['Head z'][i]))
        RightShoulderA = []
        for i in range(len(dfa)):
            RightShoulderA.append(vector(dfa['Right Shoulder x'][i], dfa['Right Shoulder y'][i], dfa['Right Shoulder z'][i]))
        RightUpperArmA = []
        for i in range(len(dfa)):
            RightUpperArmA.append(vector(dfa['Right Upper Arm x'][i], dfa['Right Upper Arm y'][i], dfa['Right Upper Arm z'][i]))
        RightForearmA = []
        for i in range(len(dfa)):
            RightForearmA.append(vector(dfa['Right Forearm x'][i], dfa['Right Forearm y'][i], dfa['Right Forearm z'][i]))
        RightHandA = []
        for i in range(len(dfa)):
            RightHandA.append(vector(dfa['Right Hand x'][i], dfa['Right Hand y'][i], dfa['Right Hand z'][i]))
        LeftShoulderA = []
        for i in range(len(dfa)):
            LeftShoulderA.append(vector(dfa['Left Shoulder x'][i], dfa['Left Shoulder y'][i], dfa['Left Shoulder z'][i]))
        LeftUpperArmA = []
        for i in range(len(dfa)):
            LeftUpperArmA.append(vector(dfa['Left Upper Arm x'][i], dfa['Left Upper Arm y'][i], dfa['Left Upper Arm z'][i]))
        LeftForearmA = []
        for i in range(len(dfa)):
            LeftForearmA.append(vector(dfa['Left Forearm x'][i], dfa['Left Forearm y'][i], dfa['Left Forearm z'][i]))
        LeftHandA = []
        for i in range(len(dfa)):
            LeftHandA.append(vector(dfa['Left Hand x'][i], dfa['Left Hand y'][i], dfa['Left Hand z'][i]))
        RightUpperLegA = []
        for i in range(len(dfa)):
            RightUpperLegA.append(vector(dfa['Right Upper Leg x'][i], dfa['Right Upper Leg y'][i], dfa['Right Upper Leg z'][i]))
        RightLowerLegA = []
        for i in range(len(dfa)):
            RightLowerLegA.append(vector(dfa['Right Lower Leg x'][i], dfa['Right Lower Leg y'][i], dfa['Right Lower Leg z'][i]))
        RightFootA = []
        for i in range(len(dfa)):
            RightFootA.append(vector(dfa['Right Foot x'][i], dfa['Right Foot y'][i], dfa['Right Foot z'][i]))
        RightToeA = []
        for i in range(len(dfa)):
            RightToeA.append(vector(dfa['Right Toe x'][i], dfa['Right Toe y'][i], dfa['Right Toe z'][i]))
        LeftUpperLegA = []
        for i in range(len(dfa)):
            LeftUpperLegA.append(vector(dfa['Left Upper Leg x'][i], dfa['Left Upper Leg y'][i], dfa['Left Upper Leg z'][i]))
        LeftLowerLegA = []
        for i in range(len(dfa)):
            LeftLowerLegA.append(vector(dfa['Left Lower Leg x'][i], dfa['Left Lower Leg y'][i], dfa['Left Lower Leg z'][i]))
        LeftFootA = []
        for i in range(len(dfa)):
            LeftFootA.append(vector(dfa['Left Foot x'][i], dfa['Left Foot y'][i], dfa['Left Foot z'][i]))
        LeftToeA = []
        for i in range(len(dfa)):
            LeftToeA.append(vector(dfa['Left Toe x'][i], dfa['Left Toe y'][i], dfa['Left Toe z'][i]))
        PelvisV = []
        for i in range(len(dfV)):
            PelvisV.append(vector(dfV['Pelvis x'][i], dfV['Pelvis y'][i], dfV['Pelvis z'][i]))
        L5V = []
        for i in range(len(dfV)):
            L5V.append(vector(dfV['L5 x'][i], dfV['L5 y'][i], dfV['L5 z'][i]))
        L3V = []
        for i in range(len(dfV)):
            L3V.append(vector(dfV['L3 x'][i], dfV['L3 y'][i], dfV['L3 z'][i]))
        T12V = []
        for i in range(len(dfV)):
            T12V.append(vector(dfV['T12 x'][i], dfV['T12 y'][i], dfV['T12 z'][i]))
        T8V = []
        for i in range(len(dfV)):
            T8V.append(vector(dfV['T8 x'][i], dfV['T8 y'][i], dfV['T8 z'][i]))
        NeckV = []
        for i in range(len(dfV)):
            NeckV.append(vector(dfV['Neck x'][i], dfV['Neck y'][i], dfV['Neck z'][i]))
        HeadV = []
        for i in range(len(dfV)):
            HeadV.append(vector(dfV['Head x'][i], dfV['Head y'][i], dfV['Head z'][i]))
        RightShoulderV = []
        for i in range(len(dfV)):
            RightShoulderV.append(vector(dfV['Right Shoulder x'][i], dfV['Right Shoulder y'][i], dfV['Right Shoulder z'][i]))
        RightUpperArmV = []
        for i in range(len(dfV)):
            RightUpperArmV.append(vector(dfV['Right Upper Arm x'][i], dfV['Right Upper Arm y'][i], dfV['Right Upper Arm z'][i]))
        RightForearmV = []
        for i in range(len(dfV)):
            RightForearmV.append(vector(dfV['Right Forearm x'][i], dfV['Right Forearm y'][i], dfV['Right Forearm z'][i]))
        RightHandV = []
        for i in range(len(dfV)):
            RightHandV.append(vector(dfV['Right Hand x'][i], dfV['Right Hand y'][i], dfV['Right Hand z'][i]))
        LeftShoulderV = []
        for i in range(len(dfV)):
            LeftShoulderV.append(vector(dfV['Left Shoulder x'][i], dfV['Left Shoulder y'][i], dfV['Left Shoulder z'][i]))
        LeftUpperArmV = []
        for i in range(len(dfV)):
            LeftUpperArmV.append(vector(dfV['Left Upper Arm x'][i], dfV['Left Upper Arm y'][i], dfV['Left Upper Arm z'][i]))
        LeftForearmV = []
        for i in range(len(dfV)):
            LeftForearmV.append(vector(dfV['Left Forearm x'][i], dfV['Left Forearm y'][i], dfV['Left Forearm z'][i]))
        LeftHandV = []
        for i in range(len(dfV)):
            LeftHandV.append(vector(dfV['Left Hand x'][i], dfV['Left Hand y'][i], dfV['Left Hand z'][i]))
        RightUpperLegV = []
        for i in range(len(dfV)):
            RightUpperLegV.append(vector(dfV['Right Upper Leg x'][i], dfV['Right Upper Leg y'][i], dfV['Right Upper Leg z'][i]))
        RightLowerLegV = []
        for i in range(len(dfV)):
            RightLowerLegV.append(vector(dfV['Right Lower Leg x'][i], dfV['Right Lower Leg y'][i], dfV['Right Lower Leg z'][i]))
        RightFootV = []
        for i in range(len(dfV)):
            RightFootV.append(vector(dfV['Right Foot x'][i], dfV['Right Foot y'][i], dfV['Right Foot z'][i]))
        RightToeV = []
        for i in range(len(dfV)):
            RightToeV.append(vector(dfV['Right Toe x'][i], dfV['Right Toe y'][i], dfV['Right Toe z'][i]))
        LeftUpperLegV = []
        for i in range(len(dfV)):
            LeftUpperLegV.append(vector(dfV['Left Upper Leg x'][i], dfV['Left Upper Leg y'][i], dfV['Left Upper Leg z'][i]))
        LeftLowerLegV = []
        for i in range(len(dfV)):
            LeftLowerLegV.append(vector(dfV['Left Lower Leg x'][i], dfV['Left Lower Leg y'][i], dfV['Left Lower Leg z'][i]))
        LeftFootV = []
        for i in range(len(dfV)):
            LeftFootV.append(vector(dfV['Left Foot x'][i], dfV['Left Foot y'][i], dfV['Left Foot z'][i]))
        LeftToeV = []
        for i in range(len(dfV)):
            LeftToeV.append(vector(dfV['Left Toe x'][i], dfV['Left Toe y'][i], dfV['Left Toe z'][i]))
        #JERK
        #jerk in x,y,z
        Pelvisx = sort(dfa['Pelvis x'])
        Pelvisy = sort(dfa['Pelvis y'])
        Pelvisz = sort(dfa['Pelvis z'])
        L5x = sort(dfa['L5 x'])
        L5y = sort(dfa['L5 y'])
        L5z = sort(dfa['L5 z'])
        L3x = sort(dfa['L3 x'])
        L3y = sort(dfa['L3 y'])
        L3z = sort(dfa['L3 z'])
        T12x = sort(dfa['T12 x'])
        T12y = sort(dfa['T12 y'])
        T12z = sort(dfa['T12 z'])
        T8x = sort(dfa['T8 x'])
        T8y = sort(dfa['T8 y'])
        T8z = sort(dfa['T8 z'])
        Neckx = sort(dfa['Neck x'])
        Necky = sort(dfa['Neck y'])
        Neckz = sort(dfa['Neck z'])
        Headx = sort(dfa['Head x'])
        Heady = sort(dfa['Head y'])
        Headz = sort(dfa['Head z'])
        RightShoulderx = sort(dfa['Right Shoulder x'])
        RightShouldery = sort(dfa['Right Shoulder y'])
        RightShoulderz = sort(dfa['Right Shoulder z'])
        RightUpperArmx = sort(dfa['Right Upper Arm x'])
        RightUpperArmy = sort(dfa['Right Upper Arm y'])
        RightUpperArmz = sort(dfa['Right Upper Arm z'])
        RightForearmx = sort(dfa['Right Forearm x'])
        RightForearmy = sort(dfa['Right Forearm y'])
        RightForearmz = sort(dfa['Right Forearm z'])
        RightHandx = sort(dfa['Right Hand x'])
        RightHandy = sort(dfa['Right Hand y'])
        RightHandz = sort(dfa['Right Hand z'])
        LeftShoulderx = sort(dfa['Left Shoulder x'])
        LeftShouldery = sort(dfa['Left Shoulder y'])
        LeftShoulderz = sort(dfa['Right Shoulder z'])
        LeftUpperArmx = sort(dfa['Left Upper Arm x'])
        LeftUpperArmy = sort(dfa['Left Upper Arm y'])
        LeftUpperArmz = sort(dfa['Left Upper Arm z'])
        LeftForearmx = sort(dfa['Left Forearm x'])
        LeftForearmy = sort(dfa['Left Forearm y'])
        LeftForearmz = sort(dfa['Left Forearm z'])
        LeftHandx = sort(dfa['Left Hand x'])
        LeftHandy = sort(dfa['Left Hand y'])
        LeftHandz = sort(dfa['Left Hand z'])
        RightUpperLegx = sort(dfa['Right Upper Leg x'])
        RightUpperLegy = sort(dfa['Right Upper Leg y'])
        RightUpperLegz = sort(dfa['Right Upper Leg z'])
        RightLowerLegx = sort(dfa['Right Lower Leg x'])
        RightLowerLegy = sort(dfa['Right Lower Leg y'])
        RightLowerLegz = sort(dfa['Right Lower Leg z'])
        RightFootx = sort(dfa['Right Foot x'])
        RightFooty = sort(dfa['Right Foot y'])
        RightFootz = sort(dfa['Right Foot z'])
        RightToex = sort(dfa['Right Toe x'])
        RightToey = sort(dfa['Right Toe y'])
        RightToez = sort(dfa['Right Toe z'])
        LeftUpperLegx = sort(dfa['Left Upper Leg x'])
        LeftUpperLegy = sort(dfa['Left Upper Leg y'])
        LeftUpperLegz = sort(dfa['Left Upper Leg z'])
        LeftLowerLegx = sort(dfa['Left Lower Leg x'])
        LeftLowerLegy = sort(dfa['Left Lower Leg y'])
        LeftLowerLegz = sort(dfa['Left Lower Leg z'])
        LeftFootx = sort(dfa['Left Foot x'])
        LeftFooty = sort(dfa['Left Foot y'])
        LeftFootz = sort(dfa['Left Foot z'])
        LeftToex = sort(dfa['Left Toe x'])
        LeftToey = sort(dfa['Left Toe y'])
        LeftToez = sort(dfa['Left Toe z'])
        #-------------------------------------------------------
        #jerk normalised
        PelvisJ = sort2(Pelvisx,Pelvisy,Pelvisz)
        L5J = sort2(L5x,L5y,L5z)
        L3J = sort2(L3x,L3y,L3z)
        T12J = sort2(T12x,T12y,T12z)
        T8J = sort2(T8x,T8y,T8z)
        NeckJ = sort2(Neckx,Necky,Neckz)
        HeadJ = sort2(Headx,Heady,Headz)
        RightShoulderJ = sort2(RightShoulderx,RightShouldery,RightShoulderz)
        RightUpperArmJ = sort2(RightUpperArmx,RightUpperArmy,RightUpperArmz)
        RightForearmJ = sort2(RightForearmx,RightForearmy,RightForearmz)
        RightHandJ = sort2(RightHandx,RightHandy,RightHandz)
        LeftShoulderJ = sort2(LeftShoulderx,LeftShouldery,LeftShoulderz)
        LeftUpperArmJ = sort2(LeftUpperArmx,LeftUpperArmy,LeftUpperArmz)
        LeftForearmJ = sort2(LeftForearmx,LeftForearmy,LeftForearmz)
        LeftHandJ = sort2(LeftHandx,LeftHandy,LeftHandz)
        RightUpperLegJ = sort2(RightUpperLegx,RightUpperLegy,RightUpperLegz)
        RightLowerLegJ = sort2(RightLowerLegx,RightLowerLegy,RightLowerLegz)
        RightFootJ = sort2(RightFootx,RightFooty,RightFootz)
        RightToeJ = sort2(RightToex,RightToey,RightToez)
        LeftUpperLegJ = sort2(LeftUpperLegx,LeftUpperLegy,LeftUpperLegz)
        LeftLowerLegJ = sort2(LeftLowerLegx,LeftLowerLegy,LeftLowerLegz)
        LeftFootJ = sort2(LeftFootx,LeftFooty,LeftFootz)
        LeftToeJ = sort2(LeftToex,LeftToey,LeftToez)
        #DISPLACEMENTS
        #Left hand - Left shoulder
        LHS = []
        for i in range(len(df)):
            LHS.append(dis(df['Left Hand x'][i], df['Left Hand y'][i], df['Left Hand z'][i], df['Left Shoulder x'][i], df['Left Shoulder y'][i], df['Left Shoulder z'][i]))
        #Right hand - Right shoulder
        RHS = []
        for i in range(len(df)):
            RHS.append(dis(df['Right Hand x'][i], df['Right Hand y'][i], df['Right Hand z'][i], df['Right Shoulder x'][i], df['Right Shoulder y'][i], df['Right Shoulder z'][i]))
        #Right hand - Left hand
        RHLH = []
        for i in range(len(df)):
            RHLH.append(dis(df['Right Hand x'][i], df['Right Hand y'][i], df['Right Hand z'][i], df['Left Hand x'][i], df['Left Hand y'][i], df['Left Hand z'][i]))
        #Head - Body Center of Mass
        HBM = []
        for i in range(len(df)):
            HBM.append(dis(df['Head x'][i], df['Head y'][i], df['Head z'][i], df2['CoM pos x'][i], df2['CoM pos y'][i], df2['CoM pos z'][i]))
        #Right Foot - Body Center of Mass
        RFBM = []
        for i in range(len(df)):
            RFBM.append(dis(df['Right Foot x'][i], df['Right Foot y'][i], df['Right Foot z'][i], df2['CoM pos x'][i], df2['CoM pos y'][i], df2['CoM pos z'][i]))
        #Left Foot - Body Center of Mass
        LFBM = []
        for i in range(len(df)):
            LFBM.append(dis(df['Left Foot x'][i], df['Left Foot y'][i], df['Left Foot z'][i], df2['CoM pos x'][i], df2['CoM pos y'][i], df2['CoM pos z'][i]))
        #Left Foot - Right Foot
        LFRF = []
        for i in range(len(df)):
            LFRF.append(dis(df['Left Foot x'][i], df['Left Foot y'][i], df['Left Foot z'][i], df['Right Foot x'][i], df['Right Foot y'][i], df['Right Foot z'][i]))
        #BOUNDING BOX/ELIPSOI
        # distance from body center
        rX = []
        for i in range(len(df)):
            rX.append(form(df['Pelvis x'][i], df['L5 x'][i], df['L3 x'][i], df['T12 x'][i], df['T8 x'][i], df['Neck x'][i], 
                        df['Head x'][i], df['Right Shoulder x'][i], df['Right Upper Arm x'][i], df['Right Forearm x'][i], df['Right Hand x'][i], 
                        df['Left Shoulder x'][i], df['Left Upper Arm x'][i], df['Left Forearm x'][i], df['Left Hand x'][i], 
                        df['Right Upper Leg x'][i], df['Right Lower Leg x'][i], df['Right Foot x'][i], df['Right Toe x'][i],
                        df['Left Upper Leg x'][i], df['Left Lower Leg x'][i], df['Left Foot x'][i], df['Left Toe x'][i]))
        rY = []
        for i in range(len(df)):
            rY.append(form(df['Pelvis x'][i], df['L5 y'][i], df['L3 y'][i], df['T12 y'][i], df['T8 y'][i], df['Neck y'][i], 
                        df['Head y'][i], df['Right Shoulder y'][i], df['Right Upper Arm y'][i], df['Right Forearm y'][i], df['Right Hand y'][i], 
                        df['Left Shoulder y'][i], df['Left Upper Arm y'][i], df['Left Forearm y'][i], df['Left Hand y'][i], 
                        df['Right Upper Leg y'][i], df['Right Lower Leg y'][i], df['Right Foot y'][i], df['Right Toe y'][i],
                        df['Left Upper Leg y'][i], df['Left Lower Leg y'][i], df['Left Foot y'][i], df['Left Toe y'][i]))
        
        rZ = []
        for i in range(len(df)):
            rZ.append(form(df['Pelvis x'][i], df['L5 z'][i], df['L3 z'][i], df['T12 z'][i], df['T8 z'][i], df['Neck z'][i], 
                        df['Head z'][i], df['Right Shoulder z'][i], df['Right Upper Arm z'][i], df['Right Forearm z'][i], df['Right Hand z'][i], 
                        df['Left Shoulder z'][i], df['Left Upper Arm z'][i], df['Left Forearm z'][i], df['Left Hand z'][i], 
                        df['Right Upper Leg z'][i], df['Right Lower Leg z'][i], df['Right Foot z'][i], df['Right Toe z'][i],
                        df['Left Upper Leg z'][i], df['Left Lower Leg z'][i], df['Left Foot z'][i], df['Left Toe z'][i]))
        #boxsides
        dX = []
        for i in range(len(rX)):
            dX.append(2*(rX[i]))
        dY = []
        for i in range(len(rX)):
            dY.append(2*(rY[i]))
        dZ = []
        for i in range(len(rX)):
            dZ.append(2*(rZ[i]))
        Rarm = []
        for i in range(len(dfa)):
            Rarm.append(sum1(RightHandA[i],RightForearmA[i],RightUpperArmA[i],RightShoulderA[i]))
        Larm = []
        for i in range(len(dfa)):
            Larm.append(sum1(LeftHandA[i],LeftForearmA[i],LeftUpperArmA[i],LeftShoulderA[i]))
        Legs = []
        for i in range(len(dfa)):
            Legs.append(sum2(LeftUpperLegA[i],LeftLowerLegA[i],LeftFootA[i],LeftToeA[i], RightUpperLegA[i],RightLowerLegA[i],RightFootA[i],RightToeA[i]))
        Body = []
        for i in range(len(dfa)):
            Body.append(sum3(RightHandA[i],RightForearmA[i],RightUpperArmA[i],RightShoulderA[i],LeftHandA[i],LeftForearmA[i],LeftUpperArmA[i],LeftShoulderA[i],
                        LeftUpperLegA[i],LeftLowerLegA[i],LeftFootA[i],LeftToeA[i], RightUpperLegA[i],RightLowerLegA[i],RightFootA[i],RightToeA[i], PelvisA[i],
                        L5A[i],L3A[i],T12A[i],T8A[i],HeadA[i],NeckA[i]))
        RarmJ = []
        for i in range(len(dfa)):
            RarmJ.append(sum1(RightHandJ[i],RightForearmJ[i],RightUpperArmJ[i],RightShoulderJ[i]))
        LarmJ = []
        for i in range(len(dfa)):
            LarmJ.append(sum1(LeftHandJ[i],LeftForearmJ[i],LeftUpperArmJ[i],LeftShoulderJ[i]))
        LegsJ = []
        for i in range(len(dfa)):
            LegsJ.append(sum2(LeftUpperLegJ[i],LeftLowerLegJ[i],LeftFootJ[i],LeftToeJ[i], RightUpperLegJ[i],RightLowerLegJ[i],RightFootJ[i],RightToeJ[i]))
        BodyJ = []
        for i in range(len(dfa)):
            BodyJ.append(sum3(RightHandJ[i],RightForearmJ[i],RightUpperArmJ[i],RightShoulderJ[i],LeftHandJ[i],LeftForearmJ[i],LeftUpperArmJ[i],LeftShoulderJ[i],
                        LeftUpperLegJ[i],LeftLowerLegJ[i],LeftFootJ[i],LeftToeJ[i], RightUpperLegJ[i],RightLowerLegJ[i],RightFootJ[i],RightToeJ[i], PelvisJ[i],
                        L5J[i],L3J[i],T12J[i],T8J[i],HeadJ[i],NeckJ[i]))
        #QOM
        RarmQ = []
        for i in range(len(dfa)):
            RarmQ.append(qom1(RightHandV[i],RightForearmV[i],RightUpperArmV[i],RightShoulderV[i]))
        LarmQ = []
        for i in range(len(dfa)):
            LarmQ.append(qom1(LeftHandV[i],LeftForearmV[i],LeftUpperArmV[i],LeftShoulderV[i]))
        LegsQ = []
        for i in range(len(dfa)):
            LegsQ.append(qom2(LeftUpperLegV[i],LeftLowerLegV[i],LeftFootV[i],LeftToeV[i], RightUpperLegV[i],RightLowerLegV[i],RightFootV[i],RightToeV[i]))
        BodyQ = []
        for i in range(len(dfa)):
            BodyQ.append(qom3(RightHandV[i],RightForearmV[i],RightUpperArmV[i],RightShoulderV[i],LeftHandV[i],LeftForearmV[i],LeftUpperArmV[i],LeftShoulderV[i],
                        LeftUpperLegV[i],LeftLowerLegV[i],LeftFootV[i],LeftToeV[i], RightUpperLegV[i],RightLowerLegV[i],RightFootV[i],RightToeV[i], PelvisV[i],
                        L5V[i],L3V[i],T12V[i],T8V[i],HeadV[i],NeckV[i]))
        #HIGH LEVEL
        #Balance
        lx = list(df2["CoM pos x"])
        ly = list(df2["CoM pos y"])
        lz = list(df2["CoM pos z"])
        CoMRx = lx[0]
        CoMRy = ly[0]
        CoMRz = lz[0]
        CoMDisp = []
        for i in range(len(df)):
            CoMDisp.append(CoMD((df2["CoM pos x"][i]), (df2["CoM pos y"][i]), (df2["CoM pos z"][i]), CoMRx, CoMRy, CoMRz))
        rX = []
        for i in range(len(df)):
            rX.append(form(df['Pelvis x'][i], df['L5 x'][i], df['L3 x'][i], df['T12 x'][i], df['T8 x'][i], df['Neck x'][i], 
                        df['Head x'][i], df['Right Shoulder x'][i], df['Right Upper Arm x'][i], df['Right Forearm x'][i], df['Right Hand x'][i], 
                        df['Left Shoulder x'][i], df['Left Upper Arm x'][i], df['Left Forearm x'][i], df['Left Hand x'][i], 
                        df['Right Upper Leg x'][i], df['Right Lower Leg x'][i], df['Right Foot x'][i], df['Right Toe x'][i],
                        df['Left Upper Leg x'][i], df['Left Lower Leg x'][i], df['Left Foot x'][i], df['Left Toe x'][i]))
        rY = []
        for i in range(len(df)):
            rY.append(form(df['Pelvis x'][i], df['L5 y'][i], df['L3 y'][i], df['T12 y'][i], df['T8 y'][i], df['Neck y'][i], 
                        df['Head y'][i], df['Right Shoulder y'][i], df['Right Upper Arm y'][i], df['Right Forearm y'][i], df['Right Hand y'][i], 
                        df['Left Shoulder y'][i], df['Left Upper Arm y'][i], df['Left Forearm y'][i], df['Left Hand y'][i], 
                        df['Right Upper Leg y'][i], df['Right Lower Leg y'][i], df['Right Foot y'][i], df['Right Toe y'][i],
                        df['Left Upper Leg y'][i], df['Left Lower Leg y'][i], df['Left Foot y'][i], df['Left Toe y'][i]))
        rZ = []
        for i in range(len(df)):
            rZ.append(form(df['Pelvis x'][i], df['L5 z'][i], df['L3 z'][i], df['T12 z'][i], df['T8 z'][i], df['Neck z'][i], 
                        df['Head z'][i], df['Right Shoulder z'][i], df['Right Upper Arm z'][i], df['Right Forearm z'][i], df['Right Hand z'][i], 
                        df['Left Shoulder z'][i], df['Left Upper Arm z'][i], df['Left Forearm z'][i], df['Left Hand z'][i], 
                        df['Right Upper Leg z'][i], df['Right Lower Leg z'][i], df['Right Foot z'][i], df['Right Toe z'][i],
                        df['Left Upper Leg z'][i], df['Left Lower Leg z'][i], df['Left Foot z'][i], df['Left Toe z'][i]))
        dX = []
        for i in range(len(rX)):
            dX.append(2*(rX[i]))
        dY = []
        for i in range(len(rX)):
            dY.append(2*(rY[i]))
        dZ = []
        for i in range(len(rX)):
            dZ.append(2*(rZ[i]))
        #Balance for elipsoid shape
        BalanceElips = []
        for i in range(len(rX)):
            Dx= abs(df['Pelvis x'][i],df2["CoM pos x"][i])
            Dy = abs(df['Pelvis y'][i],df2["CoM pos y"][i])
            Dz = abs(df['Pelvis z'][i],df2["CoM pos z"][i])
            if Dx < rX[i] and Dy < rY[i] and Dz < rZ[i]:
                BalanceElips.append(0)
            else:
                BalanceElips.append(1)
        #Balance for box shape
        BalanceBox = []
        for i in range(len(rX)):
            Dx= abs(df['Pelvis x'][i],df2["CoM pos x"][i])
            Dy = abs(df['Pelvis y'][i],df2["CoM pos y"][i])
            Dz = abs(df['Pelvis z'][i],df2["CoM pos z"][i])
            if Dx < dX[i] and Dy < dY[i] and Dz < dZ[i]:
                BalanceBox.append(0)
            else:
                BalanceBox.append(1)
        #elipsoid volume
        ElipV = []
        for i in range(len(rX)):
            ElipV.append((4/3)*(math.pi)*rX[i]*rY[i]*rZ[i])

        #bounding box volume
        BoxV = []
        for i in range(len(dX)):
            BoxV.append(dX[i]*dY[i]*dZ[i])

        #distance covered
        distP=[]
        for i in range(len(df)):
            if i==0:
                distP.append(0)
            else:
                distP.append(distance1(df["Pelvis x"][i], df["Pelvis y"][i], df["Pelvis x"][i-1], df["Pelvis y"][i-1]))
        DistP=[]
        j=0
        for i in range(0,len(distP)):
            j+=distP[i]
            DistP.append(j)
        distH=[]
        for i in range(len(df)):
            if i==0:
                distH.append(0)
            else:
                distH.append(distance1(df["Head x"][i], df["Head y"][i], df["Head x"][i-1], df["Head y"][i-1]))
        DistH=[]
        j=0
        for i in range(0,len(distH)):
            j+=distH[i]
            DistH.append(j)
        #Extensivness
        eX = []
        for i in range(len(df)):
            eX.append(form(df2['CoM pos x'][i], df['L5 x'][i], df['L3 x'][i], df['T12 x'][i], df['T8 x'][i], df['Neck x'][i], 
                        df['Head x'][i], df['Right Shoulder x'][i], df['Right Upper Arm x'][i], df['Right Forearm x'][i], df['Right Hand x'][i], 
                        df['Left Shoulder x'][i], df['Left Upper Arm x'][i], df['Left Forearm x'][i], df['Left Hand x'][i], 
                        df['Right Upper Leg x'][i], df['Right Lower Leg x'][i], df['Right Foot x'][i], df['Right Toe x'][i],
                        df['Left Upper Leg x'][i], df['Left Lower Leg x'][i], df['Left Foot x'][i], df['Left Toe x'][i]))
        eY = []
        for i in range(len(df)):
            eY.append(form(df2['CoM pos x'][i], df['L5 y'][i], df['L3 y'][i], df['T12 y'][i], df['T8 y'][i], df['Neck y'][i], 
                        df['Head y'][i], df['Right Shoulder y'][i], df['Right Upper Arm y'][i], df['Right Forearm y'][i], df['Right Hand y'][i], 
                        df['Left Shoulder y'][i], df['Left Upper Arm y'][i], df['Left Forearm y'][i], df['Left Hand y'][i], 
                        df['Right Upper Leg y'][i], df['Right Lower Leg y'][i], df['Right Foot y'][i], df['Right Toe y'][i],
                        df['Left Upper Leg y'][i], df['Left Lower Leg y'][i], df['Left Foot y'][i], df['Left Toe y'][i]))
        eZ = []
        for i in range(len(df)):
            eZ.append(form(df2['CoM pos x'][i], df['L5 z'][i], df['L3 z'][i], df['T12 z'][i], df['T8 z'][i], df['Neck z'][i], 
                        df['Head z'][i], df['Right Shoulder z'][i], df['Right Upper Arm z'][i], df['Right Forearm z'][i], df['Right Hand z'][i], 
                        df['Left Shoulder z'][i], df['Left Upper Arm z'][i], df['Left Forearm z'][i], df['Left Hand z'][i], 
                        df['Right Upper Leg z'][i], df['Right Lower Leg z'][i], df['Right Foot z'][i], df['Right Toe z'][i],
                        df['Left Upper Leg z'][i], df['Left Lower Leg z'][i], df['Left Foot z'][i], df['Left Toe z'][i]))
        Ext=[]
        for i in range(len(df)):
            Ext.append(math.sqrt(pow(eX[i],2) + pow(eY[i],2) + pow(eZ[i],2)))

        #CLASSIFICATION ------------------------------------------------------------------------------------------------------------------------------------
        #CLASSIFICATION ------------------------------------------------------------------------------------------------------------------------------------
        #create same order as trainig
        dfs = pd.DataFrame()
        dfsa = pd.DataFrame()
        dfsQ = pd.DataFrame()
        dfc = pd.DataFrame()
        dfj = pd.DataFrame()
        dfJ = pd.DataFrame()
        dfT = pd.DataFrame()
        dfTJ = pd.DataFrame()
        dfsD = pd.DataFrame()
        dfb = pd.DataFrame()
        dfe = pd.DataFrame()
        dfC_hl = pd.DataFrame()
        dfB_hl = pd.DataFrame()
        dfBV_hl = pd.DataFrame()
        dfD_hl = pd.DataFrame()
        dfE_hl = pd.DataFrame()
        dfc_hl = pd.DataFrame()
        # dfs["Pelvis"] = PelvisV
        #FLOW EFFORT
        dfTJ["Body_FE"] = BodyJ
        dfTJ["Left Arm_FE"] = LarmJ
        dfTJ["Right Arm_FE"] = RarmJ
        dfTJ["Legs_FE"] = LegsJ
        #TIME EFFORT
        dfT["Body_TE"] = Body
        # dfT["Left Arm_TE"] = Larm #missing from training dataset
        dfT["Right Arm_TE"] = Rarm
        dfT["Legs_TE"] = Legs
        #BoundingElipsoid
        dfe["Elipsoid Lenght x"] = rX
        dfe["Elipsoid Lenght y"] = rY
        dfe["Elipsoid Lenght z"] = rZ
        #BoundingBox
        dfb["Box Lenght x"] = dX
        dfb["Box Lenght y"] = dY
        dfb["Box Lenght z"] = dZ
        #displacements
        dfsD["Left Hand - Left Shoulder_D"] = LHS
        dfsD["Right Hand - Right Shoulder_D"] = RHS
        dfsD["Right Hand - Left Hand_D"] = RHLH
        dfsD["Head - Body Center of Mass_D"] = HBM
        dfsD["Right Foot - Body Center of Mass_D"] = RFBM
        dfsD["Left Foot - Body Center of Mass_D"] = LFBM
        dfsD["Left Foot - Right Foot_D"] = LFRF
        # Jerk
        dfJ["Pelvis_J"] = PelvisJ
        dfJ["L5_J"] = L5J
        dfJ["L3_J"] = L3J
        dfJ["T12_J"] = T12J
        dfJ["T8_J"] = T8J
        dfJ["Neck_J"] = NeckJ
        dfJ["Head_J"] = HeadJ
        dfJ["Right Shoulder_J"] = RightShoulderJ
        dfJ["Right Upper Arm_J"] = RightUpperArmJ
        dfJ["Right Forearm_J"] = RightForearmJ
        dfJ["Right Hand_J"] = RightHandJ
        dfJ["Left Shoulder_J"] = LeftShoulderJ
        dfJ["Left Upper Arm_J"] = LeftUpperArmJ
        dfJ["Left Forearm_J"] = LeftForearmJ
        dfJ["Left Hand_J"] = LeftHandJ
        dfJ["Right Upper Leg_J"] = RightUpperLegJ
        dfJ["Right Lower Leg_J"] = RightLowerLegJ
        dfJ["Right Foot_J"] = RightToeJ
        dfJ["Right Toe_J"] = RightToeJ
        dfJ["Left Upper Leg_J"] = LeftUpperLegJ
        dfJ["Left Lower Leg_J"] = LeftLowerLegJ
        dfJ["Left Foot_J"] = LeftToeJ
        dfJ["Left Toe_J"] = LeftToeJ
        dfj["Pelvis x_J"] = Pelvisx
        dfj["Pelvis y_J"] = Pelvisy
        dfj["Pelvis z_J"] = Pelvisz
        dfj["L5 x_J"] = L5x
        dfj["L5 y_J"] = L5y
        dfj["L5 z_J"] = L5z
        dfj["L3 x_J"] = L3x
        dfj["L3 y_J"] = L3y
        dfj["L3 z_J"] = L3z
        dfj["T12 x_J"] = T12x
        dfj["T12 y_J"] = T12y
        dfj["T12 z_J"] = T12z
        dfj["T8 x_J"] = T8x
        dfj["T8 y_J"] = T8y
        dfj["T8 z_J"] = T8z
        dfj["Neck x_J"] = Neckx
        dfj["Neck y_J"] = Necky
        dfj["Neck z_J"] = Neckz
        dfj["Head x_J"] = Headx
        dfj["Head y_J"] = Heady
        dfj["Head z_J"] = Headz
        dfj["Right Shoulder x_J"] = RightShoulderx
        dfj["Right Shoulder y_J"] = RightShouldery
        dfj["Right Shoulder z_J"] = RightShoulderz
        dfj["Right Upper Arm x_J"] = RightUpperArmx
        dfj["Right Upper Arm y_J"] = RightUpperArmy
        dfj["Right Upper Arm z_J"] = RightUpperArmz
        dfj["Right Forearm x_J"] = RightForearmx
        dfj["Right Forearm y_J"] = RightForearmy
        dfj["Right Forearm z_J"] = RightForearmz
        dfj["Right Hand x_J"] = RightHandx
        dfj["Right Hand y_J"] = RightHandy
        dfj["Right Hand z_J"] = RightHandz
        dfj["Left Shoulder x_J"] = LeftShoulderx
        dfj["Left Shoulder y_J"] = LeftShouldery
        dfj["Left Shoulder z_J"] = LeftShoulderz
        dfj["Left Upper Arm x_J"] = LeftUpperArmx
        dfj["Left Upper Arm y_J"] = LeftUpperArmy
        dfj["Left Upper Arm z_J"] = LeftUpperArmz
        dfj["Left Forearm x_J"] = LeftForearmx
        dfj["Left Forearm y_J"] = LeftForearmy
        dfj["Left Forearm z_J"] = LeftForearmz
        dfj["Left Hand x_J"] = LeftHandx
        dfj["Left Hand y_J"] = LeftHandy
        dfj["Left Hand z_J"] = LeftHandz
        dfj["Right Upper Leg x_J"] = RightUpperLegx
        dfj["Right Upper Leg y_J"] = RightUpperLegy
        dfj["Right Upper Leg z_J"] = RightUpperLegz
        dfj["Right Lower Leg x_J"] = RightLowerLegx
        dfj["Right Lower Leg y_J"] = RightLowerLegy
        dfj["Right Lower Leg z_J"] = RightLowerLegz
        dfj["Right Foot x_J"] = RightToex
        dfj["Right Foot y_J"] = RightToey
        dfj["Right Foot z_J"] = RightToez
        dfj["Right Toe x_J"] = RightToex
        dfj["Right Toe y_J"] = RightToey
        dfj["Right Toe z_J"] = RightToez
        dfj["Left Upper Leg x_J"] = LeftUpperLegx
        dfj["Left Upper Leg y_J"] = LeftUpperLegy
        dfj["Left Upper Leg z_J"] = LeftUpperLegz
        dfj["Left Lower Leg x_J"] = LeftLowerLegx
        dfj["Left Lower Leg y_J"] = LeftLowerLegy
        dfj["Left Lower Leg z_J"] = LeftLowerLegz
        dfj["Left Foot x_J"] = LeftToex
        dfj["Left Foot y_J"] = LeftToey
        dfj["Left Foot z_J"] = LeftToez
        dfj["Left Toe x_J"] = LeftToex
        dfj["Left Toe y_J"] = LeftToey
        dfj["Left Toe z_J"] = LeftToez
        #CURVATURE
        dfc["Pelvis_C"] = PelvisC
        dfc["L5_C"] = L5C
        dfc["L3_C"] = L3C
        dfc["T12_C"] = T12C
        dfc["T8_C"] = T8C
        dfc["Neck_C"] = NeckC
        dfc["Head_C"] = HeadC
        dfc["Right Shoulder_C"] = RightShoulderC
        dfc["Right Upper Arm_C"] = RightUpperArmC
        dfc["Right Forearm_C"] = RightForearmC
        dfc["Right Hand_C"] = RightHandC
        dfc["Left Shoulder_C"] = LeftShoulderC
        dfc["Left Upper Arm_C"] = LeftUpperArmC
        dfc["Left Forearm_C"] = LeftForearmC
        dfc["Left Hand_C"] = LeftHandC
        dfc["Right Upper Leg_C"] = RightUpperLegC
        dfc["Right Lower Leg_C"] = RightLowerLegC
        dfc["Right Foot_C"] = RightToeC
        dfc["Right Toe_C"] = RightToeC
        dfc["Left Upper Leg_C"] = LeftUpperLegC
        dfc["Left Lower Leg_C"] = LeftLowerLegC
        dfc["Left Foot_C"] = LeftToeC
        dfc["Left Toe_C"] = LeftToeC
        #QOM
        dfsQ["Body"] = BodyQ
        dfsQ["Left Arm"] = LarmQ
        dfsQ["Right Arm"] = RarmQ
        dfsQ["Legs"] = LegsQ
        # ACCELERATION
        dfsa["Pelvis_A"] = PelvisA
        dfsa["L5_A"] = L5A
        dfsa["L3_A"] = L3A
        dfsa["T12_A"] = T12A
        dfsa["T8_A"] = T8A
        dfsa["Neck_A"] = NeckA
        dfsa["Head_A"] = HeadA
        dfsa["Right Shoulder_A"] = RightShoulderA
        dfsa["Right Upper Arm_A"] = RightUpperArmA
        dfsa["Right Forearm_A"] = RightForearmA
        dfsa["Right Hand_A"] = RightHandA
        dfsa["Left Shoulder_A"] = LeftShoulderA
        dfsa["Left Upper Arm_A"] = LeftUpperArmA
        dfsa["Left Forearm_A"] = LeftForearmA
        dfsa["Left Hand_A"] = LeftHandA
        dfsa["Right Upper Leg_A"] = RightUpperLegA
        dfsa["Right Lower Leg_A"] = RightLowerLegA
        dfsa["Right Foot_A"] = RightToeA
        dfsa["Right Toe_A"] = RightToeA
        dfsa["Left Upper Leg_A"] = LeftUpperLegA
        dfsa["Left Lower Leg_A"] = LeftLowerLegA
        dfsa["Left Foot_A"] = LeftToeA
        dfsa["Left Toe_A"] = LeftToeA
        # VELOCITY
        dfs["Pelvis_V"] = PelvisV
        dfs["L5_V"] = L5V
        dfs["L3_V"] = L3V
        dfs["T12_V"] = T12V
        dfs["T8_V"] = T8V
        dfs["Neck_V"] = NeckV
        dfs["Head_V"] = HeadV
        dfs["Right Shoulder_V"] = RightShoulderV
        dfs["Right Upper Arm_V"] = RightUpperArmV
        dfs["Right Forearm_V"] = RightForearmV
        dfs["Right Hand_V"] = RightHandV
        dfs["Left Shoulder_V"] = LeftShoulderV
        dfs["Left Upper Arm_V"] = LeftUpperArmV
        dfs["Left Forearm_V"] = LeftForearmV
        dfs["Left Hand_V"] = LeftHandV
        dfs["Right Upper Leg_V"] = RightUpperLegV
        dfs["Right Lower Leg_V"] = RightLowerLegV
        dfs["Right Foot_V"] = RightToeV
        dfs["Right Toe_V"] = RightToeV
        dfs["Left Upper Leg_V"] = LeftUpperLegV
        dfs["Left Lower Leg_V"] = LeftLowerLegV
        dfs["Left Foot_V"] = LeftToeV
        dfs["Left Toe_V"] = LeftToeV
        #CoM-Displacement
        dfC_hl["CoMD"] = CoMDisp
        #Bounding Volume
        dfBV_hl["Bounding Elipsoid"] = ElipV
        dfBV_hl["Bounding Box"] = BoxV
        #Balance
        # dfB_hl["BalanceElipsoid"] = BalanceElips #NOT IN TRAINING
        # dfB_hl["BalanceBox"] = BalanceBox #NOT IN TRAINING
        #Distance covered
        dfD_hl["Pelvis Distance"] = DistP
        dfD_hl["Head Distance"] = DistH
        #Extensiveness
        dfE_hl["Body"] = Ext
        #Shape Directional #not in training
        # dfc_hl["Pelvis"] = shapeP
        # dfc_hl["L5"] = shapeL5
        # dfc_hl["L3"] = shapeL3
        # dfc_hl["T12"] = shapeT12
        # dfc_hl["T8"] = shapeT8
        # dfc_hl["Neck"] = shapeN
        # dfc_hl["Head"] = shapeH
        # dfc_hl["Right Shoulder"] = shapeRS
        # dfc_hl["Right Upper Arm"] = shapeRU
        # dfc_hl["Right Forearm"] = shapeRF
        # dfc_hl["Right Hand"] = shapeRH
        # dfc_hl["Left Shoulder"] = shapeLS
        # dfc_hl["Left Upper Arm"] = shapeLU
        # dfc_hl["Left Forearm"] = shapeLF
        # dfc_hl["Left Hand"] = shapeLH
        # dfc_hl["Right Upper Leg"] = shapeRUL
        # dfc_hl["Right Lower Leg"] = shapeRLL
        # dfc_hl["Right Foot"] = shapeRFO
        # dfc_hl["Right Toe"] = shapeRT
        # dfc_hl["Left Upper Leg"] = shapeLUL
        # dfc_hl["Left Lower Leg"] = shapeLLL
        # dfc_hl["Left Foot"] = shapeLFO
        # dfc_hl["Left Toe"] = shapeLT
        df_final = pd.concat((dfTJ, dfT, dfe, dfb, dfsD, dfJ, dfj, dfc, dfsQ, dfsa, dfs, dfC_hl, dfBV_hl, dfD_hl, dfE_hl), axis = 1)
        dfnp_final = df_final.to_numpy()
        dfnp_final = dfnp_final.reshape(1, 1, 200, 191)
        dfnp_final.shape
        
        y_pred = model.predict(dfnp_final)
        y_pred_2 = y_pred.argmax(axis=1)
        if y_pred_2 == 0:
            print("Low Trust")
        else:
            print("High Trust")
    
        time0 = time.time()


'''
def printDemo(x,x2):
    plt.plot(x)
    plt.plot(x2)
    plt.show()
'''
def listener():
    rospy.init_node('xsenseListener')
    rospy.Subscriber('hri_xsens_data_msg', msg_hri_all_data, callback)


    rospy.spin()


if __name__ == '__main__':
    listener()
