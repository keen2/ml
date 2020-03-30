import numpy as np
import cv2


# # Capture video from web-cam.
# video_capture = cv2.VideoCapture(0)

video_capture = cv2.VideoCapture('./shore_small.mp4')

if not video_capture.isOpened():
    print('Cannot open file or video stream.')

# ImageNet is a project which aims to provide a large image database 
# for research purposes. It contains more than 14 million images 
# which belong to more than 20,000 classes ( or synsets ).
# Read in classes of ImageNet model:
MODEL_PATH = 'model/'
with open(MODEL_PATH + 'synset_words.txt') as classes_file:
    classes = [line.strip().split(' ', maxsplit=1)[1] for line in classes_file]

# Pass the image to pre-trained model from Berkeley Vision and Learning Center:
# https://github.com/BVLC/caffe/wiki/Model-Zoo
net = cv2.dnn.readNetFromCaffe(MODEL_PATH + 'bvlc_googlenet.prototxt',
                               MODEL_PATH + 'bvlc_googlenet.caffemodel')

print('To exit press ESC.')

while True:
    ret, frame = video_capture.read()

    # From model description (.prototxt-file) we know the model takes 
    # an image of 224 by 224 as input.
    blob = cv2.dnn.blobFromImage(frame, size=(224,224))

    # Get probabilities for each class of 1000 classes:
    net.setInput(blob)
    preds = net.forward()   # [[1.66640703e-07 ... 8.46832154e-06]]
    # print(preds.shape)  # (1, 1000)

    if not ret:
        break
    
    # pos = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)//2), 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    blue_cv2 = (255, 0, 0)
    # Show top 5 classes:
    indices = np.argsort(preds[0])[-5:][::-1]
    for i, idx in enumerate(indices):
        text = '{}. {}: Probability {:.3}%'.format(i, classes[idx],
                                                   preds[0][idx] * 100)
        cv2.putText(frame, text, (20, 40*(1+i)), font, 1.0, blue_cv2, 2)
    cv2.imshow('Input stream', frame)

    # Exit if ESC was pressed. 'q' was pressed: (...& 0xFF) == ord('q')
    # It leaves only the last 8 bits of waitKey() and compares it to 27.
    if cv2.waitKey(25) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
