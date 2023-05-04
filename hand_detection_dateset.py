import cv2
import mediapipe as mp
from PIL import Image
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

FOLDERS = [
    "/home/nikodem/IVSPA/tmp1",
    "/home/nikodem/IVSPA/dislike1",
    "/home/nikodem/IVSPA/fist1",
]
# For static images:
IMAGE_FILES = []

for directory in FOLDERS:
    directory_write = directory.removesuffix("1")

    if not os.path.exists(directory_write):
        os.makedirs(directory_write)

    IMAGE_FILES.clear()
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            IMAGE_FILES.append(directory + "/" + filename)
            continue
        else:
            continue

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            print(file)
            # Print handedness and draw hand landmarks on the image.
            print("Handedness:", results.multi_handedness)

            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            # for hand_landmarks in results.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(
            #         annotated_image,
            #         hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
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
                    .landmark[mp_hands.HandLandmark.WRIST]
                    .y
                    <= results.multi_hand_landmarks[1]
                    .landmark[mp_hands.HandLandmark.WRIST]
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
                    file,
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

            # cropping image
            annotated_image = annotated_image[
                down_frame_height:up_frame_height, down_frame_width:up_frame_width
            ]

            # resizing image
            size = (256, 256)
            annotated_image = cv2.resize(
                annotated_image, size, interpolation=cv2.INTER_LINEAR
            )



            # saving image of detected hand
            cv2.imwrite(
                os.path.join(
                    directory_write, str(idx) + ".png"
                ),  # str(FOLDERS[i]) + str(idx) + '.png'),
                cv2.flip(annotated_image, 1),
            )

            print("Przetoworzno: " + IMAGE_FILES[idx], "    jako:    ", str(idx))
            continue

        print("skonczono folder: " + directory)
        print()

        # Draw hand world landmarks.
        # if not results.multi_hand_world_landmarks:
        #    continue
        # for hand_world_landmarks in results.multi_hand_world_landmarks:
        #    mp_drawing.plot_landmarks(
        #        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
















for directory in FOLDERS:
    directory = directory.removesuffix("1")
    directory_write = f"{directory}_ready"

    if not os.path.exists(directory_write):
        os.makedirs(directory_write)

    IMAGE_FILES.clear()
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            IMAGE_FILES.append(directory + "/" + filename)
            continue
        else:
            continue

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            print(file)
            # Print handedness and draw hand landmarks on the image.
            print("Handedness:", results.multi_handedness)

            if not results.multi_hand_landmarks:
                continue

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

            with open(f"{directory_write}/{idx}.txt",'w') as f:
                for number in tab:
                    f.write(f"{number};")


            print("Przetoworzno: " + IMAGE_FILES[idx], "    jako:    ", str(idx))
            continue

        print("skonczono folder: " + directory)
        print()
