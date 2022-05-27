import cv2
import time
import sys
# from inference import Network, IECore
from openvino.inference_engine import IENetwork, IECore
import numpy as np

######################### OpenVINO Initialisation #########################

# source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6
isasyncmode = False
cur_request_id = 0
next_request_id = 1

# open_vino_model = '/home/alpha/Documents/OpenVINO/models/face-detection-adas-0001.xml'
open_vino_model = './face-detection-adas-0001.xml'
open_vino_model_bin = './face-detection-adas-0001.bin'
open_vino_library = '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so'
open_vino_device = 'CPU'
open_vino_threshold = 0.8
# Instantiating the Network class
# infer_network = Network()
ie = IECore()
input_blob = None
# Calling the load_model function
# It returns the plugin and the shape of the input layer
# n,c,h,w = infer_network.load_model(open_vino_model,open_vino_device,1,1,2,open_vino_library)[1]
infer_network = ie.read_network(open_vino_model, open_vino_model_bin)
print("inputs : " + str(infer_network.inputs))
for blob_name in infer_network.inputs :
    input_blob = blob_name


out_blob = next(iter(infer_network.outputs))
print("out_blob : " + str(out_blob))

exec_net = ie.load_network(network = infer_network, device_name = "CPU")
n, c, h, w = infer_network.inputs[input_blob].shape
print(infer_network.inputs[input_blob])
print(str(n) + ", " + str(c) + ", " + str(h) +", " + str(w))
# if 1 == 1 : sys.exit(0)

############################################################################

# Using webcam feed for face detection
feed_dict = {}
cap = cv2.VideoCapture(0)
while (True):
    start_time = time.time()
    ret,frame = cap.read()
    if ret:
        # Preprocessing the frame
        frame_copy = frame.copy()
        initial_h, initial_w = frame.shape[:2]

        in_frame = cv2.resize(frame_copy, (w,h))
        in_frame = in_frame.transpose((2,0,1))
        in_frame = in_frame.reshape((n,c,h,w))

        feed_dict[input_blob] = in_frame
        res = None
        if isasyncmode:
            # infer_network.exec_net(next_request_id,in_frame)
            # exec_net(next_request_id,in_frame)
            res = exec_net.infer(inputs = feed_dict)
        else:
            # infer_network.exec_net(cur_request_id,in_frame)
            res = exec_net.infer(inputs = feed_dict)

        # if infer_network.wait(cur_request_id) == 0:
        #     res = infer_network.get_output(cur_request_id)
        res = res[out_blob]
        
            # Parsing the result
        for obj in res[0][0]:
            if obj[2] > open_vino_threshold:
                # The result obtained is normalised, hence it is being multiplied with the original width and height.
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                # cropped_image = frame_copy[ymin:ymax,xmin:xmax]
                cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                # cv2.imshow('Face',cropped_image)
                cv2.imshow("Frame", frame_copy)

                fps = 1/(time.time()-start_time)
                print('FPS : {:.2f}'.format(fps))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print('[INFO] Processing Complete ')

cap.release()
cv2.destroyAllWindows()