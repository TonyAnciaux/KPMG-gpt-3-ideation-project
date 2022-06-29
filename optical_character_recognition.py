# Literally following this tutorial: https://www.youtube.com/watch?v=vzabeKdW9tE

DEBUG = True

if DEBUG:
    from PIL import Image
    import numpy as np

    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)


DATA_DIR = "data/"
TEST_DIR = "test/"
TEST_IMAGES = DATA_DIR + "t10k-images.idx3-ubyte"
TEST_LABELS = DATA_DIR + "t10k-labels.idx1-ubyte"
TRAIN_IMAGES = DATA_DIR + "train-images.idx3-ubyte"
TRAIN_LABELS = DATA_DIR + "train-labels.idx1-ubyte"

# data_chars = "/data/archive/A_Z Handrwritten Data.csv"


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, "big")


def read_images(filename, max_n_images=None):
    images = []
    with open(filename, "rb") as file:
        _ = file.read(4)  # first 4 bytes, giving meta infos
        n_images = bytes_to_int(file.read(4))
        if max_n_images:
            n_images = max_n_images
        n_rows = bytes_to_int(file.read(4))
        n_columns = bytes_to_int(file.read(4))
        for image_index in range(n_images):
            image = []
            for row_index in range(n_rows):
                row = []
                for column_index in range(n_columns):
                    pixel = file.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, max_n_labels=None):
    labels = []
    with open(filename, "rb") as file:
        _ = file.read(4)  # first 4 bytes, giving meta infos
        n_labels = bytes_to_int(file.read(4))
        if max_n_labels:
            n_labels = max_n_labels
        for label_index in range(n_labels):
            label = file.read(1)
            labels.append(label)
    return labels


def flatten_list(data_list):
    return [pixel for sublist in data_list for pixel in sublist]


def extract_features(data_set):
    return [flatten_list(data) for data in data_set]


def dist(x, y):
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)]
    ) ** 0.5


def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_sample_index, test_sample in enumerate(X_test):
        training_distances = get_training_distances_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [pair[0] for pair in sorted(enumerate(training_distances), key=lambda x: x[1])]
        candidates = [bytes_to_int(y_train[index]) for index in sorted_distance_indices[:k]]
        top_candidate = get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    return y_pred


def main():
    X_train = read_images(TRAIN_IMAGES, 1000)
    y_train = read_labels(TRAIN_LABELS, 1000)
    X_test = read_images(TEST_IMAGES, 5)
    y_test = read_labels(TEST_LABELS, 5)

    if DEBUG:
        for index, test_sample in enumerate(X_test):
            write_image(test_sample, f"{TEST_DIR}{index}.png")

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_pred = knn(X_train, y_train, X_test, 3)

    for y_pred_i, y_test_i in zip(y_test, y_pred):
        print(y_pred_i)
        print(y_test_i)

    accuracy = sum([
        int(bytes_to_int(y_pred_i) == y_test_i)
        for y_pred_i, y_test_i
        in zip(y_test, y_pred)
    ]) / len(y_test)

    print(y_pred)
    print(f"Accuracy is {accuracy}")


if __name__ == "__main__":
    main()
