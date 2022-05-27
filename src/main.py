import time
import cv2
from face_detector import FaceDetector
from mask_detector import MaskDetector


'''
face detection code example11
'''

if __name__ == "__main__":
    face_model = './face-detection-adas-0001.xml'
    face_model_bin = './face-detection-adas-0001.bin'

    mask_model = "./face_mask.xml"
    mask_bin = "./face_mask.bin"


    open_vino_device = 'CPU'
    open_vino_threshold = 0.8
    face_detector = FaceDetector(face_model, face_model_bin, open_vino_device, open_vino_threshold)
    mask_detection = MaskDetector(mask_model, mask_bin, open_vino_device, open_vino_threshold)

    cap = cv2.VideoCapture(0)
    while(True) :
        
        start_time = time.time()

        ret,frame = cap.read()
        if ret is None : break
        frame_copy = frame.copy()

        initial_h, initial_w = frame.shape[:2]
        result = face_detector.recognize(frame_copy)
        if len(result) < 1 :
            print("continue")
            continue
        
        for obj in result :
            class_id = obj[0]
            (xmin_f, ymin_f, xmax_f, ymax_f) = obj[1:]

            margin = 20
            xmin_m = xmin_f - margin if xmin_f - margin >= 0 else 0
            ymin_m = ymin_f - margin if ymin_f - margin >= 0 else 0
            xmax_m = xmax_f + margin if xmax_f + margin <= initial_w else initial_w
            ymax_m = ymax_f + margin if ymax_f + margin <= initial_h else initial_h
            (xmin_f, ymin_f, xmax_f, ymax_f) = (xmin_m, ymin_m, xmax_m, ymax_m)

            # print(xmin_m, ", ", ymin_m, ", ", xmax_m, ", ", ymax_m)

            face = frame[ymin_m : ymax_m, xmin_m : xmax_m]
            (face_height, face_width) = face.shape[:2]
            if face_height < 20 or face_width < 20 : 
                print("face width height continue")
                continue
            mask_result = mask_detection.recognize(face)
            # mask_result = mask_detection.recognize(frame)
            mask_text = ""
            if mask_result > 0.0 :
                mask_text = "with mask (" + str(mask_result) +")"

            cv2.rectangle(frame_copy, (xmin_f, ymin_f), (xmax_f, ymax_f), (0, 255, 255), 2)
            cv2.putText(frame_copy, mask_text, (xmin_f, ymin_f - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame_copy)
        
        fps = 1/(time.time()-start_time)
        print('FPS : {:.2f}'.format(fps))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

