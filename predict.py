import cv2
import os
import sys
import numpy as np
from darkflow.net.build import TFNet

from config_options import options, base_data_dir

options = options(base_data_dir, train=False)
tfnet = TFNet(options)


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
                newImage, (top_y, top_x), (btm_y, btm_x), (0, 0, 255), 8)
            newImage = cv2.putText(newImage, label, (top_y, top_x - 5),
                                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (13, 106, 173), 3, cv2.LINE_AA)
    return newImage


def predict_on_image():
    path = options["imgdir"]
    for image in os.listdir(path):
        if image == "out":
            continue
        original_img = cv2.imread(path + image)
        # original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        results = tfnet.return_predict(original_img)
        if not results:
            continue
        print(results)
        cv2.imwrite(path + 'out/' + 'prediction_' + image, boxing(original_img, results))


def predict_on_video():
    cap = cv2.VideoCapture('GP010080.mov')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('/outputs/GP010080_preds.mov', fourcc, 20.0, (int(width), int(height)))
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame = np.asarray(frame)
            results = tfnet.return_predict(frame)
            new_frame = boxing(frame, results)
            # Display the resulting frame
            out.write(new_frame)
        else:
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_on_image()
