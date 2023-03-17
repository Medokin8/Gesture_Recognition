import cv2
import mediapipe as mp
from PIL import Image
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
#IMAGE_FILES = ["/home/nikodem/hand_detection/hand.png", "/home/nikodem/hand_detection/hand_palm.png", "/home/nikodem/hand_detection/fist.png", "/home/nikodem/hand_detection/dislike.png"]
IMAGE_FILES = []
FOLDERS=["/home/nikodem/hand_detection/tmp", "/home/nikodem/hand_detection/fist1", "/home/nikodem/hand_detection/palm1"]
for i in range(3):
    directory = FOLDERS[i]
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            IMAGE_FILES.append(directory + "/" + filename)
            #print(IMAGE_FILES)
            continue
        else:
            continue     

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            #print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            for hand_landmarks in results.multi_hand_landmarks:
                #print('hand_landmarks:', hand_landmarks)
                #print(type(hand_landmarks))
                #print(
                #    f'Index finger tip coordinates: (',
                #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                #)
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            max_height = 0
            min_height = image_height
            max_width = 0
            min_width = image_width

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width))

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width))

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width))

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width))

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width))

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width))

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width))

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width))
                                    
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width))

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width))
            
            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width))

            max_height=max(max_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height))
            min_height=min(min_height, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height))
            max_width=max(max_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width))
            min_width=min(min_width, int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width))
            
            annotated_image = annotated_image[min_height-int(0.05*min_height):max_height+int(0.05*max_height), min_width-int(0.05*min_width):max_width+int(0.05*max_width)]

            size = (256, 256)
            annotated_image = cv2.resize(annotated_image, size, interpolation = cv2.INTER_LINEAR)

            cv2.imwrite(
                os.path.join(str(FOLDERS[i]),
                str(idx) + '.png'), #str(FOLDERS[i]) + str(idx) + '.png'),
                cv2.flip(annotated_image, 1))

            print("Przetoworzno: " + str(idx))
            continue
        
        print()
        print("skonczono folder: " + directory)
            
            
            
            # Draw hand world landmarks.
            #if not results.multi_hand_world_landmarks:
            #    continue
            #for hand_world_landmarks in results.multi_hand_world_landmarks:
            #    mp_drawing.plot_landmarks(
            #        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
