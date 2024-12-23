import cv2
import numpy as np
#rom skimage import color, filters, morphology
import matplotlib.pyplot as plt
import mediapipe as mp
# Charger l'image
image_path = 'image 1.jpg'
image = cv2.imread(image_path)

# 1. Prétraitement - Conversion en niveaux de gris
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Segmentation - Seuillage adaptatif pour isoler la main
#_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(gray_image, 40, 150)
#binary_image = cv2.bitwise_not(binary_image)

# Affichage de l'image binaire
plt.figure(figsize=(8, 8))
plt.imshow(edges, cmap="gray")
plt.title("Image segmentée ")
plt.axis("off")
plt.show()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    results = hands.process(edges)

    # Vérifie si une main a été détectée
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les points clés et les connexions sur l'image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Affichage des coordonnées des points clés (optionnel)
            for idx, landmark in enumerate(hand_landmarks.landmark):
                print(f"Point {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")

            # Affichage de l'image annotée
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.title("Squelette de la main avec MediaPipe")
            plt.axis("off")
            plt.show()
    else:
        print("Aucune main détectée dans l'image.")