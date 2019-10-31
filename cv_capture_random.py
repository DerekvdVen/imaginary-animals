# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import setup_path 
import airsim
import pprint
import tempfile
import os
import time
import random
import itertools

# Variables #
date = "2019-11"
area_x = 500
area_y = 500

pp = pprint.PrettyPrinter(indent=4)

client = airsim.VehicleClient()

airsim.wait_key('Press any key to get camera parameters')
for camera_id in range(2):
    camera_info = client.simGetCameraInfo(str(camera_id))
    print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))

airsim.wait_key('Press any key to get images')
#tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
tmp_dir = "C:/Users/Derek/Documents/Thesis_python/Data/images/" + date
tmp_dir2 = "C:/Users/Derek/Documents/Thesis_python/Data/semantic/" + date
print ("Saving images to %s" % tmp_dir)

try:
    os.makedirs(os.path.join(tmp_dir))
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

try:
    os.makedirs(os.path.join(tmp_dir2))
except OSError:
    if not os.path.isdir(tmp_dir2):
        raise
#client.simSetTimeOfDay(True, start_datetime = "2018-10-14 {}:12:00")
Z = -(int(input("how high? (m)")))
S = int(input("How big is the stepsize?"))
px = 0 
py = 0
counter = 0
while counter < 1000:
    x1 = random.randrange(-area_x,area_x,S)
    y1 = random.randrange(-area_y,area_y,S)
    fill_value = x1
    distance_y = abs(y1 - py)
    distance_x = abs(x1 - px)
    if distance_y < distance_x:
        fill_value = y1
    print("Flying to:",x1,y1)
    print("current_location = ",px,",",py)
    time.sleep(3)
    S1,S2 = S, S
    if x1 < px:
        S1 = -S
    if y1 < py:
        S2 = -S
    for x, y in itertools.zip_longest(range(px, x1, S1), range(py, y1, S2), fillvalue = fill_value): # do few times
        counter += 1
        #xn = 1 + x*5  # some random number
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, Z), airsim.to_quaternion(-1.57079633, 0, 0)), True)
        time.sleep(0.5)
        #print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))
        print(x,'___',y)
    
        angles = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
        #print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene)])
        responses_s = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Segmentation)])

        for i, response in enumerate(responses):
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            airsim.write_file(os.path.normpath(os.path.join(tmp_dir,  "IMG" + str(counter) + '_' + str(x) + ',' + str(y) + '.png')), response.image_data_uint8)
        for i, response in enumerate(responses_s):
            print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            airsim.write_file(os.path.normpath(os.path.join(tmp_dir2,  "IMG" + str(counter) + '_' + str(x) + ',' + str(y) + '.png')), response.image_data_uint8)
            
        pose = client.simGetVehiclePose()
        #pp.pprint(pose)
    px = x1
    py = y1
    time.sleep(3)

# currently reset() doesn't work in CV mode. Below is the workaround
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-1.57079633, 0, 0)), True)
