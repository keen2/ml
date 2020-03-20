import numpy as np
import face_recognition
import cv2

# This is a script running face recognition on a video file 
# and saving the results to a new video file.

# Open the input movie file.
video_capture = cv2.VideoCapture('short_hamilton_clip.mp4')
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output file (resolution/frame rate should match input video!).
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_movie = cv2.VideoWriter('output.mp4', fourcc, 29.97, (640, 360))

# Load sample pictures and learn how to recognize it.
obama_1 = face_recognition.load_image_file('obama.jpg')
obama_1_face_encoding = face_recognition.face_encodings(obama_1)[0]
obama_2 = face_recognition.load_image_file('obama2.jpg')
obama_2_face_encoding = face_recognition.face_encodings(obama_2)[0]

# Load a sample picture of second person and learn how to recognize it.
lin_image = face_recognition.load_image_file('lin-manuel-miranda.png')
lin_face_encoding = face_recognition.face_encodings(lin_image)[0]

# Create arrays of known face encodings and their names.
known_face_encodings = [obama_1_face_encoding,
                        obama_2_face_encoding,
                        lin_face_encoding]
known_face_names = ['Barack Obama',
                    'Barack Obama',
                    'Lin-Manuel Miranda']

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video.
    ret, frame = video_capture.read()
    frame_number += 1

    # Exit when the video file ends.
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) 
    # to RGB color (which face_recognition uses).
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = rgb_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video.
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s).
        matches = face_recognition.compare_faces(known_face_encodings,
                                                 face_encoding, tolerance=0.5)
        name = 'Unknown'

        # # a)
        # # If a match was found in known encodings, use the first one.
        # if True in matches:
        #     name = known_face_names[matches.index(True)]

        # b) 
        # Or instead, use the known face with the smallest distance 
        # to the new face.
        distances = face_recognition.face_distance(known_face_encodings,
                                                   face_encoding)
        best_match_index = np.argmin(distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)


    # Display the results.
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face.
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
                      (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0,
                    (255, 255, 255), 1)

    # Write the resulting image to the output video file.
    if frame_number % 15 == 0:
        print('Writing frame {} / {}'.format(frame_number, length))
    output_movie.write(frame)

# Release handle of input file.
video_capture.release()
cv2.destroyAllWindows()
