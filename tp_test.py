import cv2
import numpy as np


def display_convexity_defects(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))
    image_copy = image.copy()  # Copy for drawing

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological transformations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assuming it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the convex hull and defects
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(largest_contour[s][0])  # Start point of the defect
                end = tuple(largest_contour[e][0])  # End point of the defect
                far = tuple(largest_contour[f][0])  # Point farthest from the hull

                # Draw the defect points and lines
                cv2.line(image_copy, start, end, (0, 255, 0), 2)  # Line between start and end
                cv2.circle(image_copy, start, 5, (255, 0, 0), -1)  # Start point
                cv2.circle(image_copy, end, 5, (0, 255, 255), -1)  # End point
                cv2.circle(image_copy, far, 5, (0, 0, 255), -1)  # Far point (defect)

    # Display the original image, mask, and results
    cv2.imshow("Original Image", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Convexity Defects", image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the function
display_convexity_defects('image 1.jpg')  # Replace with your image path
