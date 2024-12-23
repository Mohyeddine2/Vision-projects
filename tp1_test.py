import cv2
import numpy as np
image = cv2.imread('image 5.jpg')
image = cv2.resize(image, (640, 480))
image_copy = image.copy()

    # Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a skin color range (adjust based on your lighting conditions)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask for skin color
mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological transformations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=6 )

    # Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color
    cv2.drawContours(mask, [contour], -1, color, 2)
cv2.imshow("Hand Number", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
if contours:
        # Find the largest contour (assuming it's the hand)
    largest_contour = max(contours, key=cv2.contourArea)

        # Draw the contour on the original image
    cv2.drawContours(image_copy, [largest_contour], -1, (0, 255, 0), 2)

        # Find convex hull
    hull = cv2.convexHull(largest_contour, returnPoints=fals)
    cv2.drawContours(image_copy, [hull], -1, (0, 255, 0), 2)

        # Detect convexity defects
    defects = cv2.convexityDefects(largest_contour, hull)

    finger_count = 0  # Initialize finger count

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(largest_contour[s][0])
            end = tuple(largest_contour[e][0])
            far = tuple(largest_contour[f][0])

                # Calculate the angle between start, far, and end points
            a = np.linalg.norm(np.array(start) - np.array(far))
            b = np.linalg.norm(np.array(end) - np.array(far))
            c = np.linalg.norm(np.array(start) - np.array(end))

            angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

                # Count as a finger if the angle is less than 90 degrees and the depth is significant
            if angle < np.pi / 2 and d > 10000:
                finger_count += 1

        # Include the thumb by adding 1 to the finger count
    finger_count += 1

        # Display the detected number
    cv2.putText(image_copy, f"Number: {finger_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the results
   # cv2.imshow("Original Image", image)
    #cv2.imshow("HSV",hsv)
cv2.imshow("Mask", mask)
cv2.imshow("Hand Number", image_copy)



