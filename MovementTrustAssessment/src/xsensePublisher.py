#!/usr/bin/env python3
"""

Protocol manual available at:
https://www.xsens.com/hubfs/Downloads/Manuals/MVN_real-time_network_streaming_protocol_specification.pdf


Prop read has not been tested.
Mind that the ID is tthe segment index +1 as stated in page 13 of the manual.
ID 24 is blank
"""

import socket
import netifaces as ni
import struct
import time
from os import system, name
import rospy
from custom_msg_hri.msg import msg_hri_mass, msg_hri_multi, msg_hri_all_data

# Header values
dType =  0          # 2 ascii bytes      , Type of datagram (tick on mvn)                     
sCount = 0          # 32-bit unsigned int, 
dCount = 0          # 32-bit unsigned int, first bit (128) means is the last datagram
nItem  = 0          #  8-bit unsigned int
tCode  = 0          # 32-bit unsigned int
nSeg   = 0          #  8-bit int
nProp  = 0          #  8-bit int

# segment values
seg_quaternions = {"PosX":0,"PosY":0,"PosZ":0,"Qre":0,"Qi":0,"Qj":0,"Qk":0 }
seg_eurler      = {"PosX":0,"PosY":0,"PosZ":0,"RotX":0,"RotY":0,"RotZ":0}

seg_lin_kin     = {"PosX":0,"PosY":0,"PosZ":0,"VelX":0,"VelY":0,"VelZ":0,
                   "AccX":0,"AccY":0,"AccZ":0}
seg_ang_kin     = {"Qre":0,"Qi":0,"Qj":0,"Qk":0,"AVelX":0,"AVelY":0,"AVelZ":0,
                   "AAccX":0,"AAccY":0,"AAccZ":0}


dic_segments ={1:{"name":"name", "data":""},2:{"name":"name", "data":""},3:{"name":"name", "data":""},4:{"name":"name", "data":""},5:{"name":"name", "data":""},6:{"name":"name", "data":""},
               7:{"name":"name", "data":""},8:{"name":"name", "data":""},9:{"name":"name", "data":""},10:{"name":"name", "data":""},11:{"name":"name", "data":""},12:{"name":"name", "data":""},
               13:{"name":"name", "data":""},14:{"name":"name", "data":""},15:{"name":"name", "data":""},16:{"name":"name", "data":""},17:{"name":"name", "data":""},18:{"name":"name", "data":""},
               19:{"name":"name", "data":""},20:{"name":"name", "data":""},21:{"name":"name", "data":""},22:{"name":"name", "data":""},23:{"name":"name", "data":""}, 24:{}, 25:{"name":"name", "data":""},
               26:{"name":"name", "data":""},27:{"name":"name", "data":""},28: {"name":"name", "data":""}}

dic_mass    = {"CoMX":"","CoMY":"","CoMZ":""}


seg_all     = {"PosX":0,"PosY":0,"PosZ":0,"RotX":0,"RotY":0,"RotZ":0,           # Pos and Eurler
               "Qre":0,"Qi":0,"Qj":0,"Qk":0,                                    # Quaternions
               "VelX":0,"VelY":0,"VelZ":0,"AccX":0,"AccY":0,"AccZ":0,           # Lin Kinematics
               "AVelX":0,"AVelY":0,"AVelZ":0,"AAccX":0,"AAccY":0,"AAccZ":0}     # Ang Kinematics

names       = ["Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head", "Right Shoulder"
                , "Right Upper Arm", "Right Forearm", "Right Hand", "Left Shoulder"
                , "Left Upper Arm", "left Forearm", "Left Hand", "Right Upper Leg", "Right Lower Leg"
                , "Right Foot", "Right Toe", "Left Upper Leg", "Left Lower Leg"
                , "Left Foot", "Left Toe", "None", "Prop1", "Prop2", "Prop3", "Prop4"]
def initializeNames():

    for i in range(28):
        dic_segments[i+1]["name"]= names[i]
        dic_segments[i+1]["data"]= seg_all
        
initializeNames()

def readHeader():
    global data
    global dType, sCount, dCount, nItem, tCode, nSeg, nProp
    
    dType  = int(data[4:6])
    sCount = int.from_bytes(bytearray(data[6:10]),'big')
    dCount = int.from_bytes(bytearray(data[10:11]),'big')
    nItem  = int.from_bytes(bytearray(data[11:12]),'big')
    tCode  = int.from_bytes(bytearray(data[12:16]),'big')
    nSeg   = int.from_bytes(bytearray(data[17:18]),'big')
    nProp  = int.from_bytes(bytearray(data[18:19]),'big')
    #print("FIRST PRINT")
    #print(F"nItem  {nItem}")
    """
    print(F"dType  {dType}")
    print(F"sCount {sCount}")
    print(F"dCount {dCount}")
    
    print(F"tCode {tCode}")
    print(F"nSeg {nSeg}")
    print(F"nProp {nProp}")
    """
    if(dCount < 128):
        print("ERROR, increase the MTU  and buffer size. The payload is splited "
              "in more than one datagrams, this code does not handle it")


def readQuaternion(pos):
    initBit = pos*32 +24
    
    """
    print(F"ID {int.from_bytes(bytearray(data[initBit:initBit+4]),'big')}")
    print(f"PosX {struct.unpack('>f',bytearray(data[initBit+4 : initBit+8 ]))}")
    print(f"PosY {struct.unpack('>f',bytearray(data[initBit+8 : initBit+12]))}")
    print(f"PosZ {struct.unpack('>f',bytearray(data[initBit+12: initBit+16]))}")
    print(f"Qre  {struct.unpack('>f',bytearray(data[initBit+16: initBit+20]))}")
    print(f"Qi   {struct.unpack('>f',bytearray(data[initBit+20: initBit+24]))}")
    print(f"Qj   {struct.unpack('>f',bytearray(data[initBit+24: initBit+28]))}")
    print(f"Qk   {struct.unpack('>f',bytearray(data[initBit+28: initBit+32]))}")
    """
    ID   = int.from_bytes(bytearray(data[initBit:initBit+4]),'big')
    PosX = struct.unpack('>f',bytearray(data[initBit+4 : initBit+8 ]))[0]
    PosY = struct.unpack('>f',bytearray(data[initBit+8 : initBit+12]))[0]
    PosZ = struct.unpack('>f',bytearray(data[initBit+12: initBit+16]))[0]
    Qre  = struct.unpack('>f',bytearray(data[initBit+16: initBit+20]))[0]
    Qi   = struct.unpack('>f',bytearray(data[initBit+20: initBit+24]))[0]
    Qj   = struct.unpack('>f',bytearray(data[initBit+24: initBit+28]))[0]
    Qk   = struct.unpack('>f',bytearray(data[initBit+28: initBit+32]))[0]
    
    seg_quaternions = {"PosX":PosX,"PosY":PosY,"PosZ":PosZ,"Qre":Qre,"Qi":Qi,"Qj":Qj,"Qk":Qk }    
    dic_segments[ID]["data"].update(seg_quaternions)
    

    
def readEuler(pos):
    initBit = pos*28 +24
    """
    print(F"ID {int.from_bytes(bytearray(data[initBit:initBit+4]),'big')}")
    print(f"PosX {struct.unpack('>f',bytearray(data[initBit+4 : initBit+8 ]))}")
    print(f"PosY {struct.unpack('>f',bytearray(data[initBit+8 : initBit+12]))}")
    print(f"PosZ {struct.unpack('>f',bytearray(data[initBit+12: initBit+16]))}")
    print(f"Rx   {struct.unpack('>f',bytearray(data[initBit+16: initBit+20]))}")
    print(f"Ry   {struct.unpack('>f',bytearray(data[initBit+20: initBit+24]))}")
    print(f"Rz   {struct.unpack('>f',bytearray(data[initBit+24: initBit+28]))}")
    """
    ID   = int.from_bytes(bytearray(data[initBit:initBit+4]),'big')
    PosX = struct.unpack('>f',bytearray(data[initBit+4 : initBit+8 ]))[0]
    PosY = struct.unpack('>f',bytearray(data[initBit+8 : initBit+12]))[0]
    PosZ = struct.unpack('>f',bytearray(data[initBit+12: initBit+16]))[0]    
    Rx   = struct.unpack('>f',bytearray(data[initBit+16: initBit+20]))[0]
    Ry   = struct.unpack('>f',bytearray(data[initBit+20: initBit+24]))[0]
    Rz   = struct.unpack('>f',bytearray(data[initBit+24: initBit+28]))[0]
    
    seg_eurler = {"PosX":PosX,"PosY":PosY,"PosZ":PosZ,"RotX":Rx,"RotY":Ry,"RotZ":Rz}
    dic_segments[ID]["data"].update(seg_eurler)
    


def readAngularKinematics(pos):
    initBit = pos*44 +24

    ID   = int.from_bytes(bytearray(data[initBit:initBit+4]),'big')
    Qre  = struct.unpack('>f',bytearray(data[initBit+4 : initBit+8 ]))[0]
    Qi   = struct.unpack('>f',bytearray(data[initBit+8 : initBit+12]))[0]
    Qj   = struct.unpack('>f',bytearray(data[initBit+12: initBit+16]))[0]    
    Qk   = struct.unpack('>f',bytearray(data[initBit+16: initBit+20]))[0]
    AVelX = struct.unpack('>f',bytearray(data[initBit+20: initBit+24]))[0]
    AVelY = struct.unpack('>f',bytearray(data[initBit+24: initBit+28]))[0]
    AVelZ = struct.unpack('>f',bytearray(data[initBit+28: initBit+32]))[0]
    AAccX = struct.unpack('>f',bytearray(data[initBit+32: initBit+36]))[0]
    AAccY = struct.unpack('>f',bytearray(data[initBit+36: initBit+40]))[0]
    AAccZ = struct.unpack('>f',bytearray(data[initBit+40: initBit+44]))[0]
    
    
    seg_ang_kin     = {"Qre":Qre,"Qi":Qi,"Qj":Qj,"Qk":Qk,"AVelX":AVelX,
                       "AVelY":AVelY,"AVelZ":AVelY,"AAccX":AAccX,"AAccY":AAccY,
                       "AAccZ":AAccZ}
    
    dic_segments[ID]["data"].update(seg_ang_kin)

    """
    if(ID == 1):
        print(f"Qre {Qre:.3f}\tQi {Qi:.3f}\tQj {Qj:.3f}\tQk {Qk:.3f}")
        print(f"AVelX {AVelX:.3f}\tAVelY {AVelY:.3f}\tAVelZ {AVelZ:.3f}")
        print(f"AAccX {AAccX:.3f}\tAAccY {AAccY:.3f}\tAAccZ {AAccZ:.3f}\n")
    """
    
    
def readLinearKinematics(pos):
    initBit = pos*40 +24

    ID   = int.from_bytes(bytearray(data[initBit:initBit+4]),'big')
    PosX = struct.unpack('>f',bytearray(data[initBit+4 : initBit+8 ]))[0]
    PosY = struct.unpack('>f',bytearray(data[initBit+8 : initBit+12]))[0]
    PosZ = struct.unpack('>f',bytearray(data[initBit+12: initBit+16]))[0]    
    VelX = struct.unpack('>f',bytearray(data[initBit+16: initBit+20]))[0]
    VelY = struct.unpack('>f',bytearray(data[initBit+20: initBit+24]))[0]
    VelZ = struct.unpack('>f',bytearray(data[initBit+24: initBit+28]))[0]
    AccX = struct.unpack('>f',bytearray(data[initBit+28: initBit+32]))[0]
    AccY = struct.unpack('>f',bytearray(data[initBit+32: initBit+36]))[0]
    AccZ = struct.unpack('>f',bytearray(data[initBit+36: initBit+40]))[0]
    
    
    
    seg_lin_Kin = {"PosX":PosX,"PosY":PosY,"PosZ":PosZ,"VelX":VelX,"VelY":VelY,
                   "VelZ":VelZ, "AccX":AccX,"AccY":AccY,"AccZ":AccZ}
    
    dic_segments[ID]["data"].update(seg_lin_Kin)
    """
    if(ID == 1):
        print(f"PosX {PosX:.3f}\tPosY {PosY:.3f}\tPosZ {PosZ:.3f}")
        print(f"VelX {VelX:.3f}\tVelY {VelY:.3f}\tVelZ {VelZ:.3f}")
        print(f"AccX {AccX:.3f}\tAccY {AccY:.3f}\tAccZ {AccZ:.3f}\n")
    """
def readCenterOfMass():
    global dic_mass
    initBit = 24
    massX = struct.unpack('>f',bytearray(data[initBit   : initBit+4 ]))[0]
    massY = struct.unpack('>f',bytearray(data[initBit+4 : initBit+8 ]))[0]
    massZ = struct.unpack('>f',bytearray(data[initBit+8 : initBit+12]))[0]   

    #print(f"Center of mass X {massX:.3f}, Y {massY:.3f}, Z , {massZ:.3f}")
    dic_mass = {"CoMX":massX, "CoMY":massY, "CoMZ":massZ}


def readData():
    global dType
    def readTypePos(p):
        if dType == 1:      readEuler(p)
        elif dType == 2:    readQuaternion(p)
        elif dType == 21:   readLinearKinematics(p)
        elif dType == 22:   readAngularKinematics(p)

    
    if dType == 24:
        readCenterOfMass()
    elif(dType in [1,2,21,22]): # If dType is supported then read it    
        for i in range(nSeg):
            readTypePos(i)        
        if nProp != 0:          # If there is props then read them (not tested)
            for i in range(nProp):
                readTypePos(i+24)        
    else:
        print("Read data ERROR, wrong Type of Message. Check your Stream Settings")
        print(f"Data Type = {dType}")


def printConsole(datagrams, fps):
    
    global dType, sCount, dCount, nItem, tCode, nSeg, nProp
    
    # Clear console
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')
    print(f"Interface Running");


    if 1  in datagrams: print("Data Type      \t\t= EULER")
    if 2  in datagrams: print("Data Type      \t\t= QUATERNION")
    if 21 in datagrams: print("Data Type      \t\t= Linear Kinematics")
    if 22 in datagrams: print("Data Type      \t\t= Angular Kinematics")
    if 24 in datagrams: print("Data Type      \t\t= Center of Mass")
    #else: print(f"ID = {dType} -Error, wrong Type of Message. Check your Stream Settings")
        
    print(f"Number of Sample    \t= {sCount}")
    print(f"Number of Datagram  \t= {dCount}")
    print(f"Number of Items     \t= {nItem}")
    print(f"Time Code           \t= {tCode}")
    print(f"Number of Segments  \t= {nSeg}")
    print(f"Number of Props     \t= {nProp}")
    
    print(f"------- aprox FPS {fps} -------") 


def main():
    global data
    iface = ni.interfaces()[3]
    ip_address = nip = ni.ifaddresses(iface)[ni.AF_INET][0]['addr']
    UDP_IP = ip_address
    #UDP_IP = "172.24.211.234"  # IP of the own PC
    UDP_PORT = 9763

    print(f"\n\nConnecting to netInface  {iface} ip = {UDP_IP}")
    print("If stuck here, \n\t1.Check Xsens is publishing to this ip")
    print("\t2.Revise the code in line 273 of this script")
    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.bind((UDP_IP, UDP_PORT))
    data, addr = sock.recvfrom(4096)  # buffer size is 1024 bytes

    reads = 0
    prevDatagrams = []      # Count the datagrams types per frame
    prevSample = 0          # Keep track of the frame
    dic_lastData = dic_segments.copy() # saved after the frame is read 
    timer = time.time()

    readHeader()
    prevSample = sCount
    printConsole([""], "")

    pub = rospy.Publisher('hri_xsens_data_msg', msg_hri_all_data, queue_size=10)
    msg = msg_hri_all_data()
    rospy.init_node('hri_xsens_data')

    while (True):
        data, addr = sock.recvfrom(4096) # buffer size is 1024 bytes
        readHeader()
        #print("New pack")

        if ( prevSample< sCount and prevSample-2 > sCount): # If you get a lost package from the last 2 frames discard it
            print("Received a lost package, it is discarded")
            print(f"prev = {prevSample} and sCount{sCount}")
        else:
            #print(f"not dicarded     {prevSample}       {sCount}")
            if(prevSample == sCount-1 or  prevSample > sCount+24):
                reads +=1
                #print(timer)
                if timer <= time.time() - 2:    # Each 2 sec refresh the console
                    printConsole(prevDatagrams, reads)
                    reads = 0
                    timer += 1
                prevSample = sCount
                prevDatagrams = []
                dic_data_lastframe = dic_segments.copy()




                # Header
                msg.stamp          = rospy.Time.now()
                msg.frame_id       = 'hri-xsese-msg'
                msg.sample_counter = sCount

                # Center of mass
                msg.center_of_mass.center_of_mass.x = dic_mass["CoMX"]
                msg.center_of_mass.center_of_mass.y = dic_mass["CoMY"]
                msg.center_of_mass.center_of_mass.z = dic_mass["CoMZ"]

                for i in range(nItem):
                    msg_multi = msg_hri_multi()
                    msg_multi.joint_name = dic_data_lastframe[i+1]["name"]

                    msg_multi.position.x = dic_data_lastframe[i+1]["data"]["PosX"]
                    msg_multi.position.y = dic_data_lastframe[i+1]["data"]["PosY"]
                    msg_multi.position.z = dic_data_lastframe[i+1]["data"]["PosZ"]

                    msg_multi.euler.x = dic_data_lastframe[i+1]["data"]["RotX"]
                    msg_multi.euler.y = dic_data_lastframe[i+1]["data"]["RotY"]
                    msg_multi.euler.z = dic_data_lastframe[i+1]["data"]["RotZ"]

                    msg_multi.quaternion.x = dic_data_lastframe[i+1]["data"]["Qre"]
                    msg_multi.quaternion.y = dic_data_lastframe[i+1]["data"]["Qi"]
                    msg_multi.quaternion.z = dic_data_lastframe[i+1]["data"]["Qj"]
                    msg_multi.quaternion.w = dic_data_lastframe[i+1]["data"]["Qk"]

                    msg_multi.velocity.x = dic_data_lastframe[i+1]["data"]["VelX"]
                    msg_multi.velocity.y = dic_data_lastframe[i+1]["data"]["VelY"]
                    msg_multi.velocity.z = dic_data_lastframe[i+1]["data"]["VelZ"]

                    msg_multi.acceleration.x = dic_data_lastframe[i+1]["data"]["AccX"]
                    msg_multi.acceleration.y = dic_data_lastframe[i+1]["data"]["AccY"]
                    msg_multi.acceleration.z = dic_data_lastframe[i+1]["data"]["AccZ"]

                    msg_multi.ang_velocity.x = dic_data_lastframe[i+1]["data"]["AVelX"]
                    msg_multi.ang_velocity.y = dic_data_lastframe[i+1]["data"]["AVelY"]
                    msg_multi.ang_velocity.z = dic_data_lastframe[i+1]["data"]["AVelZ"]

                    msg_multi.ang_acceleration.x = dic_data_lastframe[i+1]["data"]["AAccX"]
                    msg_multi.ang_acceleration.y = dic_data_lastframe[i+1]["data"]["AAccY"]
                    msg_multi.ang_acceleration.z = dic_data_lastframe[i+1]["data"]["AAccZ"]
                    msg.joints_data.append(msg_multi)
                    
                    
                    


                pub.publish(msg)
                


            """
                Your code goes here.
                to refer to each segment you can use -> dic_data_lastframe[ID]["data"]["property"]
                
                You can access the header values using the global vairables
                
            """
            	
            
            
            if(prevSample == sCount):           # while the frame is the same read more datagrams
                if(dType in prevDatagrams):     # If it is received multiple of the same datagram for same frame (MVN stopped)  
                    print("MVN Stopped")        # Force the loop to run the code again, but with same data as MVN is stopped.             
                    prevDatagrams = []
                else:
                    prevDatagrams.append(dType)
                    readData()       
                    
if __name__ == "__main__":
    main()

seg_all     = {"PosX":0,"PosY":0,"PosZ":0,"RotX":0,"RotY":0,"RotZ":0,           # Pos and Eurler
               "Qre":0,"Qi":0,"Qj":0,"Qk":0,                                    # Quaternions
               "VelX":0,"VelY":0,"VelZ":0,"AccX":0,"AccY":0,"AccZ":0,           # Lin Kinematics
               "AVelX":0,"AVelY":0,"AVelZ":0,"AAccX":0,"AAccY":0,"AAccZ":0}
