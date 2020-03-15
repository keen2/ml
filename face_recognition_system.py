__author__ = "Andrei Ermishin"
__copyright__ = "Copyright (c) 2020"
__license__ = "GNU GPLv3"
__email__ = "andrey.yermishin@gmail.com"


import face_recognition


# Load the known images
num_known = 3
known_face_encodings = []
for idx in range(num_known):
    image = face_recognition.load_image_file(f'person_{idx+1}.jpg')
    
    # Get the face encoding of each person.
    encodings = face_recognition.face_encodings(image)
    # This can fail if no one is found in the photo. Take first person.
    known_face_encodings.append(encodings[0] if encodings else None)

# Load images we want to check.
num_unknown = 8
unknown_face_encodings = []
for idx in range(num_unknown):
    unknown_image = face_recognition.load_image_file(f'unknown_{idx+1}.jpg')

    width, height, _rgb = unknown_image.shape

    # 'unknown_7.jpg' has low resolution and algorithm can't find Person 2.
    # Tuning for low resolution images:
    if width + height < 500:
        face_locations = face_recognition.face_locations(unknown_image,
                                        number_of_times_to_upsample=2)
        unknown_face_encodings = face_recognition.face_encodings(unknown_image,
                                        known_face_locations=face_locations)
    else:
        # Get face encodings for any people in the picture.
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    # There might be more than one person in the photo, 
    # so we need to loop over each face we found.
    for unknown_face in unknown_face_encodings:
        # Test if the unknown face encoding matches any of the 3 people we know
        matches = face_recognition.compare_faces(known_face_encodings,
                                                 unknown_face)

        for match_idx in range(len(matches)):
            if matches[match_idx]:
                print(f'Found Person {match_idx+1} in the photo #{idx+1}!')
