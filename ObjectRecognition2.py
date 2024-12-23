import cv2
import numpy as np
from mypy.messages import best_matches

images=[]
for i in range(5):
    path = "image "+str(i+1)+".jpg"
    image = cv2.imread(path)
    images.append(image)
def calculate_hu_moments(images):
    moments_list = []
    for image in images:
        smoothed_img = cv2.GaussianBlur(image, (3, 3), 0)
        resized_img = cv2.resize(smoothed_img, (318, 513))
        if len(smoothed_img.shape) == 3 and smoothed_img.shape[2] == 3:
            resized_img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        else:
            resized_img_gray = smoothed_img
        mask = np.zeros_like(resized_img_gray, dtype=np.uint8)
        mask[-100:, :] = 255
        resized_img_gray = cv2.bitwise_or(resized_img_gray, mask)
        _, binary = cv2.threshold(resized_img_gray, 127, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(binary)
        huMoments = cv2.HuMoments(moments)

        for i in range(len(huMoments)):
            if huMoments[i] != 0:
                huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(abs(huMoments[i]))
            else:
                huMoments[i] = 0
        moments_list.append(huMoments)
    return moments_list
def compare_hu_moments(moment_known, moment_unknown):
    distances = {
        'Euclidean': [],
        'Manhattan': [],
        'Chebyshev': []
    }
    for moments1 in moment_known:
        euclidean_distance = np.sqrt(np.sum((moments1 - moment_unknown) ** 2))
        manhattan_distance = np.sum(np.abs(moments1 - moment_unknown))
        chebyshev_distance = np.max(np.abs(moments1 - moment_unknown))
        distances['Euclidean'].append(euclidean_distance)
        distances['Manhattan'].append(manhattan_distance)
        distances['Chebyshev'].append(chebyshev_distance)
    return distances
def best_matching(moment_known, moment_unknown):
    distances = compare_hu_moments(moment_known, moment_unknown)
    best_matches = {}
    for metric, values in distances.items():
        best_index = np.argmin(values)
        min_distance = values[best_index]
        best_matches[metric] = {
            "index": best_index,
            "distance": min_distance,
            "moments": moment_known[best_index]
        }
        print(metric, best_index, "signe : ",best_index+1 )
    return best_matches
moments_list = calculate_hu_moments(images)
img = cv2.imread("image 2.jpg")
moment_unknown = calculate_hu_moments(img)
distances = compare_hu_moments(moments_list, moment_unknown)

best_matches = best_matching(moments_list, moment_unknown)
#cv2.putText(img, f"Number: {index}", (20, 20),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

cv2.imshow('image', img)
cv2.imshow('image unkown',images[3])

cv2.waitKey(0)
cv2.destroyAllWindows()


