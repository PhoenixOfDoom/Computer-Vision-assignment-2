import cv2
import numpy as np
import Tester
import AssistantFunctions
from scipy.ndimage import minimum_filter
import matplotlib.pyplot as plt


# Calculate the disparity maps (from both directions) by the following local stereo algorithm.
# the all-takes-winner method after aggregation on volume-cost which is calculated on the basis of
# :consistency filter with census-transform
#  Calculate transform-census on each of the images
#  Calculate the volume-cost (note that the maximum disparity is given in the txt file).
#  Perform local aggregation
#  Find the minimum for each location
#  Filter matches based on test-c

def Calculate_Disparty_Map(LEFT_IMG, RIGHT_IMG,max_disparity=134):
    LEFT_IMG_Census = AssistantFunctions.census_transform(LEFT_IMG)
    RIGHT_IMG_Census = AssistantFunctions.census_transform(RIGHT_IMG)

    Volume_Cost = AssistantFunctions.compute_volume_cost(LEFT_IMG,RIGHT_IMG,134)

    # print(Volume_Cost)

    aggregate_local = AssistantFunctions.compute_aggregate_local(Volume_Cost,window_size=9)

    # print(aggregate_local)

    disparity_map = np.argmin(Volume_Cost, axis=2)

    Left_Disparity = disparity_map
    Right_Disparity = np.argmin(Volume_Cost.transpose(1, 0, 2), axis=2)


    # cv2.imshow("LEFT Transformed Image", LEFT_IMG_Census)
    # cv2.imshow("RIGHT Transformed Image", RIGHT_IMG_Census)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return Left_Disparity,Right_Disparity

# Calculate the two depth maps from it.



# For the left image: calculate a reprojection of all image coordinates (pixels) to
# The three-dimensional space. This can be done using the matrix K of the intrinsics (the inverse) and the map
# The depths D. The points in 3D will be in the coordinate system of the camera, meaning that we
# We use the camera matrix [��|��]�� = �� where �� = �� , 0 = �� .


# Drop the 3D points back onto the left camera plane, in its original position. now,
# Synthesize the resulting image by copying the pixel values in RGB from the original image.
# The result should be the same as the original image (to the point of holes) - this is in total a correctness check of
# 2D-3D-2D the answer


# Repeat for a series of 11 camera positions, at uniform 1 cm intervals on the
# baseline between the two images. This can be done by manipulating T in the extrinsic matrix.

# ==================================================================================================================
def DO_Assignment():
    Matrix = Tester.Extract_K("data/set_1/K.txt")
    # Left_IMG = cv2.imread("data/set_1/im_left.jpg")
    # Right_IMG = cv2.imread("data/set_1/im_right.jpg")
    max_disparity = Tester.Extract_MaxDisp("data/set_1/max_disp.txt")

    Left_IMG = cv2.imread("data/set_1/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    RIGHT_IMG = cv2.imread("data/set_1/im_right.jpg", cv2.IMREAD_GRAYSCALE)

    Left_Disparity,Right_Disparity = Calculate_Disparty_Map(Left_IMG, RIGHT_IMG)

    # Create subplots for left image, left disparity map, right image, and right disparity map
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    # Left image and disparity map
    axs[0, 0].imshow(cv2.cvtColor(Left_IMG, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Left Image')
    axs[0, 1].imshow(Left_Disparity, cmap='gray')
    axs[0, 1].set_title('Left Disparity Map')

    # Right image and disparity map
    axs[1, 0].imshow(cv2.cvtColor(RIGHT_IMG, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Right Image')
    axs[1, 1].imshow(Right_Disparity, cmap='gray')
    axs[1, 1].set_title('Right Disparity Map')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    return














# ==================================================================================================================
# CHAT-GPT


