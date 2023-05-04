import joblib
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier

classifier: RandomForestClassifier = joblib.load('test.pkl')

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error opening video stream or file")
# else:
#     cv2.namedWindow("Pose Classification", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Pose Classification", 1280, 720)

#     while cap.isOpened():
#         # Capture frame-by-frame. If frame is read correctly, proceed with movenet model procedures.
#         ret, frame = cap.read()
#         if ret:

frame=cv2.imread('fist1/6.png')

with mp.solutions.hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
) as hands:
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        exit()
    image_height, image_width, _ = frame.shape
    annotated_image = frame.copy()

    # for hand_landmarks in results.multi_hand_landmarks:
    #     mp_drawing.draw_landmarks(
    #         annotated_image,
    #         hand_landmarks,
    #         mp.solutions.hands.Hands.HAND_CONNECTIONS,
    #         mp_drawing_styles.get_default_hand_landmarks_style(),
    #         mp_drawing_styles.get_default_hand_connections_style())

    max_height = 0
    min_height = image_height
    max_width = 0
    min_width = image_width

    if len(results.multi_hand_landmarks) >= 2:
        print("if dla pliku dwie rece: ", file)
        if (
            results.multi_hand_landmarks[0]
            .landmark[mp.solutions.hands.Hands.HandLandmark.WRIST]
            .y
            <= results.multi_hand_landmarks[1]
            .landmark[mp.solutions.hands.Hands.HandLandmark.WRIST]
            .y
        ):
            for idx_hand in range(21):
                max_height = max(
                    max_height,
                    int(
                        results.multi_hand_landmarks[0].landmark[idx_hand].y
                        * image_height
                    ),
                )
                min_height = min(
                    min_height,
                    int(
                        results.multi_hand_landmarks[0].landmark[idx_hand].y
                        * image_height
                    ),
                )
                max_width = max(
                    max_width,
                    int(
                        results.multi_hand_landmarks[0].landmark[idx_hand].x
                        * image_width
                    ),
                )
                min_width = min(
                    min_width,
                    int(
                        results.multi_hand_landmarks[0].landmark[idx_hand].x
                        * image_width
                    ),
                )
        else:
            for idx_hand in range(21):
                max_height = max(
                    max_height,
                    int(
                        results.multi_hand_landmarks[1].landmark[idx_hand].y
                        * image_height
                    ),
                )
                min_height = min(
                    min_height,
                    int(
                        results.multi_hand_landmarks[1].landmark[idx_hand].y
                        * image_height
                    ),
                )
                max_width = max(
                    max_width,
                    int(
                        results.multi_hand_landmarks[1].landmark[idx_hand].x
                        * image_width
                    ),
                )
                min_width = min(
                    min_width,
                    int(
                        results.multi_hand_landmarks[1].landmark[idx_hand].x
                        * image_width
                    ),
                )

    else:
        print(
            "else dla pliku:  ",
            #file,
            "   o dlug: ",
            len(results.multi_hand_landmarks),
        )
        hand_landmarks = results.multi_hand_landmarks[0]
        for idx_hand in range(21):
            max_height = max(
                max_height,
                int(hand_landmarks.landmark[idx_hand].y * image_height),
            )
            min_height = min(
                min_height,
                int(hand_landmarks.landmark[idx_hand].y * image_height),
            )
            max_width = max(
                max_width,
                int(hand_landmarks.landmark[idx_hand].x * image_width),
            )
            min_width = min(
                min_width,
                int(hand_landmarks.landmark[idx_hand].x * image_width),
            )

    # to avoid cropping below and above image
    down_frame_height = min_height - int(0.05 * min_height)
    up_frame_height = max_height + int(0.05 * max_height)
    if down_frame_height <= 0:
        down_frame_height = 0
    if up_frame_height >= image_height:
        up_frame_height = image_height

    down_frame_width = min_width - int(0.05 * min_width)
    up_frame_width = max_width + int(0.05 * max_width)
    if down_frame_width <= 0:
        down_frame_width = 0
    if up_frame_width >= image_width:
        up_frame_width = image_width

    annotated_image = annotated_image[
        down_frame_height:up_frame_height, down_frame_width:up_frame_width
    ]

    results = hands.process(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    tab = []
    for landmark in  results.multi_hand_landmarks[0].landmark:
        tab.append(
            landmark.x
        )
        tab.append(
            landmark.y
        )
        tab.append(
            landmark.z
        )
        
    result = classifier.predict(np.expand_dims(tab, axis=0))
    print(result)

    for hand_landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())
        
    cv2.imwrite('test.png', annotated_image)

# cv2.imshow("Pose Classification", frame)    









