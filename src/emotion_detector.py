"""Detect the main facial emotion in an image using DeepFace."""
import cv2
from deepface import DeepFace


class EmotionDetector:
    def detect(self, image_bgr):
        """Return (emotion, scores, face_region), or (None, {}, None) if no face is found.

        scores = DeepFace's confidence for each of the 7 emotions, in % (they add up to 100).
        """
        try:
            faces = DeepFace.analyze(
                image_bgr,
                actions=["emotion"],
                detector_backend="opencv",  # OpenCV Haar cascade finds the face
                enforce_detection=True,      # raise an error instead of guessing on a face-less image
                silent=True,
            )
        except ValueError as error:
            if "Face could not be detected" in str(error):  # DeepFace's "no face" error
                return None, {}, None
            raise  # any other problem (e.g. broken install) should show up, not look like "no face"

        # If several faces are found, use the biggest one (closest to the camera).
        face = max(faces, key=lambda f: f["region"]["w"] * f["region"]["h"])
        scores = {emotion: float(score) for emotion, score in face["emotion"].items()}
        return face["dominant_emotion"], scores, face["region"]

    @staticmethod
    def draw_box(image_bgr, region):
        """Draw a green box around the face DeepFace found (no second detector needed)."""
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        boxed = image_bgr.copy()
        thickness = max(2, image_bgr.shape[1] // 150)  # thicker lines on bigger photos
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (84, 185, 29), thickness)  # Spotify green, in BGR
        return boxed
