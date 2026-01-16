'''
To install face_recognition, simply use 'pip install face_recognition' in a terminal
However, often you may meet an error about the 'dlib' library with cmake.
The easy solution is to visit https://github.com/z-mahmud22/Dlib_Windows_Python3.x and download the 
compiled wheels locally with the python version, and install it from local
'''

import cv2
import time
import face_recognition
import os

def save_found_image(unknown_image, face_locations_to_draw, output_folder="faces_detected"):
    """
    Draws bounding boxes and "Match" label on TOP of the face (not covering the face)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for top, right, bottom, left in face_locations_to_draw:
        # Draw the green bounding box around the face
        cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 3)

        # Create a filled rectangle for the "Match" label at the TOP of the box
        label_height = 35
        cv2.rectangle(unknown_image, (left, top - label_height), (right, top), (0, 255, 0), cv2.FILLED)

        # Put the "Match" text inside that top rectangle
        cv2.putText(unknown_image, "Match", (left + 6, top - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Convert from RGB (face_recognition) to BGR (OpenCV save)
    image_bgr = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

    timestamp = int(time.time())
    filename = f"matched_face_{timestamp}.jpg"
    output_path = os.path.join(output_folder, filename)

    success = cv2.imwrite(output_path, image_bgr)
    if success:
        print(f"Match found and saved to: {output_path}")
    else:
        print("Failed to save image.")

start = time.time()

# Load known face
known_image = face_recognition.load_image_file("known_man.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

folder_path = "imageset/"
filenames = [file.name for file in os.scandir(folder_path) if file.is_file()]

for filename in filenames:
    full_path = os.path.join(folder_path, filename)
    unknown_image = face_recognition.load_image_file(full_path)

    # Detect faces and get encodings efficiently
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    matched_locations = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)[0]
        if match:
            matched_locations.append(face_location)

    if matched_locations:
        print(f"Match found in {filename} ({len(matched_locations)} match(es))")
        image_to_save = unknown_image.copy()
        save_found_image(image_to_save, matched_locations)

print("Total time:", time.time() - start)