import AssignmentSolution
import Tester
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
def RunTester(FolderName):
    pictures, K_Value, MaxDisp_Value = Tester.Extract_TestFiles(FolderName=FolderName)
    return pictures, K_Value, MaxDisp_Value

# def RunOutputer(FolderName):
#     picture_left , picture_right;
#     return;

def Main():
    folder_name = "data/set_" + str(1) + "/"
    pictures, K_Value, MaxDisp_Value = RunTester(folder_name)
    print(pictures)
    print(K_Value)
    print(MaxDisp_Value)

    image_Left_URL = folder_name + pictures[0]
    image_Right_URL = folder_name + pictures[1]

    print(image_Left_URL)
    print(image_Right_URL)

    image_Left = plt.imread(image_Left_URL)
    image_Right = plt.imread(image_Right_URL)

    # Convert the image to grayscale
    image_Left_GS = cv2.cvtColor(image_Left, cv2.COLOR_BGR2GRAY)

    # Convert the image to grayscale
    image_Right_GS = cv2.cvtColor(image_Right, cv2.COLOR_BGR2GRAY)

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Display the first image with subtitle
    ax1.imshow(image_Left)
    ax1.set_title("image Left")

    # Display the second image with subtitle
    ax2.imshow(image_Right)
    ax2.set_title("image Right")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    # TODO
    plt.show()
    # Start the timer
    start_time = time.time()

    AssignmentSolution.DO_Assignment()
    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    Main();


