import face_recognition
import numpy as np


class Classification():
    def __init__(self) -> None:
        biden_image = face_recognition.load_image_file("biden.png")

        self.known_face_encodings = [face_recognition.face_encodings(biden_image)[0]]
        self.known_face_names = ['Joe Biden']

    def classify(self, image: np.ndarray):
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)
        return face_names
    