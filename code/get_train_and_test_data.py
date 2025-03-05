from load_images import load_images_from_folder
import numpy as np

# Import the load_images function from load_images.py

# Use the load_images function to get the training and testing data
def get_train_and_test_data():
    train_indoor_images_path = 'dataset/Training/museum-indoor'
    train_outdoor_images_path = 'dataset/Training/museum-outdoor'
    test_indoor_images_path = 'dataset/Testing/museum-indoor'
    test_outdoor_images_path = 'dataset/Testing/museum-outdoor'

    train_indoor_images, train_indoor_labels = load_images_from_folder(train_indoor_images_path, 0)  # Label 0 for indoor
    train_outdoor_images, train_outdoor_labels = load_images_from_folder(train_outdoor_images_path, 1)  # Label 1 for outdoor

    # Load testing images
    test_indoor_images, test_indoor_labels = load_images_from_folder(test_indoor_images_path, 0)  # Label 0 for indoor
    test_outdoor_images, test_outdoor_labels = load_images_from_folder(test_outdoor_images_path, 1)  # Label 1 for outdoor

    # Combine training images and labels
    x_train = train_indoor_images + train_outdoor_images
    y_train = train_indoor_labels + train_outdoor_labels

    # Combine testing images and labels
    x_test = test_indoor_images + test_outdoor_images
    y_test = test_indoor_labels + test_outdoor_labels

    print("length of train imahes",len(x_train))
    print("length of test imahes",len(x_test))
    print("length of train labels",len(y_train))
    print("length of test labels",len(y_test))
    # Convert lists to NumPy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    get_train_and_test_data()
    print("Data loaded successfully !!!")