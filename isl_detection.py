import copy
import itertools
import string

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow import keras

# Load the saved model once at import time (safe; no webcam side-effects)
model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list))) or 1.0

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def run_detection(camera_index: int = 0, window_name: str = 'Indian sign language detector', stop_event=None):
    """Run live ISL detection loop using webcam.

    Press ESC to exit the detection window. If a stop_event (threading.Event) is provided,
    the loop will terminate when stop_event.is_set() is True.
    """
    cap = cv2.VideoCapture(camera_index)
    try:
        with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while cap.isOpened():
                # Allow external shutdown
                if stop_event is not None and getattr(stop_event, "is_set", None) and stop_event.is_set():
                    break

                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Flip the image horizontally for a selfie-view display.
                image = cv2.flip(image, 1)

                # To improve performance, optionally mark the image as not writeable to pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                debug_image = copy.deepcopy(image)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        # Draw the landmarks
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
                        df = pd.DataFrame(pre_processed_landmark_list).transpose()

                        # predict the sign language
                        predictions = model.predict(df, verbose=0)
                        # get the predicted class for each sample
                        predicted_classes = np.argmax(predictions, axis=1)
                        label = alphabet[predicted_classes[0]]
                        cv2.putText(
                            image,
                            label,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            2,
                        )
                        print(label)
                        print("------------------------")

                # output image
                cv2.imshow(window_name, image)
                # Exit on ESC or external stop
                if (cv2.waitKey(5) & 0xFF) == 27:
                    break
                if stop_event is not None and getattr(stop_event, "is_set", None) and stop_event.is_set():
                    break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def generate_frames(camera_index: int = 0, window_name: str = 'Indian sign language detector', stop_event=None):
    """Generator that yields JPEG-encoded video frames with ISL annotations.

    This is suitable for use with a Django StreamingHttpResponse to render
    the camera stream inline on a web page via an <img> tag pointing to
    a multipart/x-mixed-replace endpoint.
    """
    cap = cv2.VideoCapture(camera_index)
    try:
        with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while cap.isOpened():
                if stop_event is not None and getattr(stop_event, "is_set", None) and stop_event.is_set():
                    break

                success, image = cap.read()
                if not success:
                    continue

                image = cv2.flip(image, 1)
                image.flags.writeable = False
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                image.flags.writeable = True
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                debug_image = copy.deepcopy(image)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
                        df = pd.DataFrame(pre_processed_landmark_list).transpose()
                        predictions = model.predict(df, verbose=0)
                        predicted_classes = np.argmax(predictions, axis=1)
                        label = alphabet[predicted_classes[0]]
                        cv2.putText(
                            image,
                            label,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            2,
                        )
                ret, buffer = cv2.imencode('.jpg', image)
                if not ret:
                    continue
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    run_detection()
