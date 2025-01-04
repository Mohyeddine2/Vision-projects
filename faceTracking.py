import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Initialisation des paramètres
N = 100
process_noise = [5, 5, 2, 2, 3,3]  # Bruits pour [x, y, w, h, vx, vy]
observation_noise = 10
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# Détecter le visage initial
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
if len(faces) > 0:
    x, y, w, h = faces[0]
    particles = np.random.normal([x + w // 2, y + h // 2, w, h, 0, 0], process_noise, (N, 6))
    weights = np.ones(N) / N
else:
    print("Aucun visage détecté !")
    cap.release()
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Si un visage est détecté, on met à jour les particules
    if len(faces) > 0:
        x, y, w, h = faces[0]
        particles = np.random.normal([x + w // 2, y + h // 2, w, h, 0, 0], process_noise, (N, 6))
        weights = np.ones(N) / N
        observation = [x + w // 2, y + h // 2, w, h]
    else:
        observation = None
    # Propagation des particules
    particles[:, :2] += particles[:, 4:]  # Mise à jour de (x, y) avec (v_x, v_y)
    particles[:, :4] += np.random.normal(0, process_noise[:4], particles[:, :4].shape)  # Ajout du bruit
    # Mise à jour des poids si un visage est détecté
    if observation is not None:
        distances = np.linalg.norm(particles[:, :2] - observation[:2], axis=1)
        weights = np.exp(-0.5 * (distances ** 2) / observation_noise ** 2)
        weights += 1e-300
        weights /= np.sum(weights)
    else:
        # Si aucun visage n'est détecté, on propague les particules sans mettre à jour les poids
        weights.fill(1.0 / N)

    # Rééchantillonnage
    indices = np.random.choice(range(N), size=N, p=weights)
    particles = particles[indices]
    weights.fill(1.0 / N)
    # Estimation par calcule de moyenne
    estimated_state = np.mean(particles, axis=0).astype(int)
    x_est, y_est, w_est, h_est = estimated_state[:4]
    cv2.rectangle(frame, (x_est - w_est // 2, y_est - h_est // 2),
                  (x_est + w_est // 2, y_est + h_est // 2), (255, 0, 0), 2)
    cv2.imshow("Suivi de Visage", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
