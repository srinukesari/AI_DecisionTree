import os
import cv2
import numpy as np

def load_images_from_folder(folder_path,label):
    images = []
    labels = []
    folder_path = os.path.join('/Users/srinukesari/IdeaProjects/Comp6721_AI_Assigments/AI_DecisionTree/', folder_path)
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            # Resize the image to a consistent size
            img = cv2.resize(img, (64, 64))  # Example: Resize to 64x64 pixels
            # Flatten the image into a 1D array
            img = img.flatten()
            images.append(img)
            labels.append(label)  # Assign the specified label to each image
    # print(images)
    return images, labels
