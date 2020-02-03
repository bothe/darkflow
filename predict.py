import cv2
import os
import json
import numpy as np
from darkflow.net.build import TFNet


def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.5:
            newImage = cv2.rectangle(
                newImage, (top_y, top_x), (btm_y, btm_x), (173, 106, 13), 8)
            newImage = cv2.putText(newImage, label, (top_y, top_x-5),
                                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 255, 0), 3, cv2.LINE_AA)

    return newImage


options = {"model": "cfg/custom-yolov2.cfg",
           "load": -1,
           "gpu": 1.0}
tfnet = TFNet(options)
tfnet.load_from_ckpt()
for image in os.listdir('images/'):
    original_img = cv2.imread("images/"+image)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet.return_predict(original_img)
    print(results)
    cv2.imwrite('/valohai/outputs/'+'prediction' +
                image, boxing(original_img, results))
