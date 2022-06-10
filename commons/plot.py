from matplotlib import pylab as plt

def print2D(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def printXandY(out, label):
    plt.subplot(1, 2, 1)
    plt.imshow(out, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='gray')
    plt.show()