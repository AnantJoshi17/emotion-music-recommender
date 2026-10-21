"""Measure DeepFace's emotion accuracy on the FER-2013 test set.

1. Download FER-2013 (image folders) from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
2. Unzip it so you have  fer2013/test/<emotion>/*.jpg
3. Run:  python evaluate.py fer2013/test --per-class 100

It takes the same number of images from each emotion (a balanced sample), runs DeepFace on
each one and prints accuracy, per-emotion precision/recall and a confusion matrix.
"""
import argparse
import random
from pathlib import Path

import cv2
import pandas as pd
from deepface import DeepFace
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", help="folder with one sub-folder per emotion")
    parser.add_argument("--per-class", type=int, default=100, help="images per emotion")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    y_true, y_pred = [], []
    for emotion in EMOTIONS:
        files = sorted((Path(args.test_dir) / emotion).glob("*.jpg"))
        random.Random(args.seed).shuffle(files)
        for path in files[: args.per_class]:
            image = cv2.imread(str(path))
            # FER-2013 images are already cropped faces, so skip face detection.
            result = DeepFace.analyze(image, actions=["emotion"], detector_backend="skip",
                                      enforce_detection=False, silent=True)
            y_true.append(emotion)
            y_pred.append(result[0]["dominant_emotion"])
        print(f"{emotion:>8}: {len(files[: args.per_class])} images done")

    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.1%} on {len(y_true)} images\n")
    print(classification_report(y_true, y_pred, labels=EMOTIONS, zero_division=0))
    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=EMOTIONS),
                          index=[f"true {e}" for e in EMOTIONS], columns=EMOTIONS)
    print("Confusion matrix (rows = true emotion, columns = predicted):")
    print(matrix.to_string())


if __name__ == "__main__":
    main()
