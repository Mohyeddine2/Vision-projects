import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def calculate_histogram(image, bins=32):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()
def intersection_hist(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
def bhattacharyya_distance(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
def minkowski_distance(hist1, hist2, p=2):
    return np.sum(np.abs(hist1 - hist2) ** p) ** (1 / p)
def matusita_distance(hist1, hist2):
    return np.sqrt(np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2))
def cosine_distance(hist1, hist2):
    cosine_similarity = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
    return cosine_similarity
def emd(hist1, hist2):
    bins1 = np.array([[i, hist1[i]] for i in range(len(hist1))], dtype=np.float32)
    bins2 = np.array([[i, hist2[i]] for i in range(len(hist2))], dtype=np.float32)
    emd_value, _, _ = cv2.EMD(bins1, bins2, cv2.DIST_L2)
    return emd_value
def find_minimum_distance(reference_image_path, scene_folder, bins=32):
    # Charger l'image de référence
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {reference_image_path}")
    reference_hist = calculate_histogram(reference_image, bins)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    min_distances = {
        'Bhattacharyya': float('inf'),
        'Minkowski': float('inf'),
        'Matusita': float('inf'),
        'Cosine': float('inf'),
        'EMD': float('inf')
    }
    closest_images = {
        'Bhattacharyya': None,
        'Minkowski': None,
        'Matusita': None,
        'Cosine': None,
        'EMD': None
    }
    for filename in os.listdir(scene_folder):
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(scene_folder, filename)
            target_image = cv2.imread(file_path)
            if target_image is None:
                print(f"Erreur : Impossible de charger l'image {file_path}")
                continue
            target_hist = calculate_histogram(target_image, bins)
            distances = {
                'Bhattacharyya': bhattacharyya_distance(reference_hist, target_hist),
                'Minkowski': minkowski_distance(reference_hist, target_hist),
                'Matusita': matusita_distance(reference_hist, target_hist),
                'Cosine': cosine_distance(reference_hist, target_hist),
                'EMD': emd(reference_hist, target_hist)
            }

            for metric, distance in distances.items():
                if distance < min_distances[metric]:
                    min_distances[metric] = distance
                    closest_images[metric] = file_path
    return min_distances, closest_images, reference_image,distances
def visualize_results(reference_image, closest_images, min_distances):
    metrics = list(closest_images.keys())
    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image de Référence")
    axes[0].axis('off')
    for i, metric in enumerate(metrics):
        closest_image = cv2.imread(closest_images[metric])
        if closest_image is not None:
            axes[i + 1].imshow(cv2.cvtColor(closest_image, cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(f"{metric}\n{min_distances[metric]:.4f}")
        else:
            axes[i + 1].set_title(f"{metric}\nImage non trouvée")
        axes[i + 1].axis('off')
    plt.tight_layout()
    plt.show()
reference_image_path = "Scene1/158.jpg"
scene_folder = "Scene1"
min_distances, closest_images, reference_image, dis = find_minimum_distance(reference_image_path, scene_folder)
print("Distances minimales et images correspondantes :")
for metric, distance in min_distances.items():
    print(f"{metric} : {distance:.4f} (Image : {closest_images[metric]})")
visualize_results(reference_image, closest_images, min_distances)

