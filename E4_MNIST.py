import struct, os
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


def load_mnist(image_file, label_file, path="."):
    digits = np.arange(10)

    fname_image = os.path.join(path, image_file)
    fname_label = os.path.join(path, label_file)

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((1, rows * cols))
        labels[i] = lbl[ind[i]]

    return images, labels


if __name__ == "__main__":
    train_image, train_label = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    test_image, test_label = load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    # train_image = [im / 255.0 for im in train_image]
    # train_label = [im / 255.0 for im in train_label]
    # test_image = [im / 255.0 for im in test_image]
    # test_label = [im / 255.0 for im in test_label]
    knc = KNeighborsClassifier(n_neighbors=10)
    knc.fit(train_image, train_label)
    predict = knc.predict(test_image)
    print("accuracy_score: %.4lf" % accuracy_score(predict, test_label))
    print("Classification report for classifier %s:\n%s\n" % (knc, classification_report(imgtesttarget, predict)))
