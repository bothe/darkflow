import os

import cv2
import numpy as np

from config_options import options, base_data_dir
from darkflow.net.build import TFNet

options = options(base_data_dir, train=False)
tfnet = TFNet(options)
tfnet.predict()


def boxing(original_img, box):
    if box["confidence"] > options["threshold"]:
        original_img = cv2.rectangle(original_img,
                                     (box['topleft']['x'], box['topleft']['y']),
                                     (box['bottomright']['x'], box['bottomright']['y']),
                                     box['color'], box["thick"] // 2)
        original_img = cv2.putText(original_img, box['label'],
                                   (box['topleft']['x'], box['topleft']['y'] - 12), 0,
                                   1e-3 * original_img.shape[0], box['color'], int(box["thick"] // 3))
        return original_img


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
        for box in results:
            boxing(original_img, box)
        cv2.imwrite(path + 'out/' + 'prediction_' + image, original_img)


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
            for box in results:
                new_frame = boxing(frame, box)
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
