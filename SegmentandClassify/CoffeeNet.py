import cv2 as cv
import os
import matplotlib.pyplot as plt
from Classifier.k_means import KMEANS
from Classifier.ycgcr import YCGCR
from Classifier.ImageSegment import ImageSegment


class CoffeeNet:
    # Path to saved model
    model_path = '/home/frexg/Keras_Practice/models/CoffeeNet_V2_MobileNet'
    test_image_directory = '/home/frexg/Downloads/lara2018-master/segmentation/dataset/test_binary_sens'

    def getImagePath(self, image_path, filename):
        # get path of image file
        return os.path.join(image_path, filename)

    def getModelPath(self):
        # get path of model
        return self.model_path

    def classifyImage(self, imagepath, filename):
        # Start Image Segmentation
        imageName = filename.split('.')[0]
        print(imageName)
        ground_truth = cv.imread(os.path.join(
            self.test_image_directory, f'{imageName}_mask.png'))

        # HThresh, inp, HTime, Hacc = ImageSegment(self.getImagePath(
        #    imagepath, filename), self.getModelPath(), filename).getThresh()

        # print("IoU YCgCr")
        YThresh, YTime, Yacc = YCGCR(filename, image=self.getImagePath(
            image_path, filename)).getThresh()
        # KThresh, KTime, Kacc = KMEANS(filename, image=self.getImagePath(
        #    image_path, filename)).getThresh()

        """ plt.imshow(inp)
        plt.title("Input")
        plt.show()

        plt.imshow(ground_truth, cmap="gray")
        plt.title("Ground Truth")
        plt.show()

        plt.imshow(HThresh, cmap="gray")
        plt.title("Robust HSV")
        plt.show()

        plt.imshow(YThresh, cmap="gray")
        plt.title("YCgCr")
        plt.show()

        plt.imshow(KThresh, cmap="gray")
        plt.title("K-means")
        plt.show()
        """
        """
        fig, axs = plt.subplots(2, 3)
        axs[0][0].imshow(inp)
        axs[0][0].set_title("Input")

        axs[0][1].imshow(ground_truth, cmap="gray")
        axs[0][1].set_title("Ground Truth")

        axs[0][2].imshow(HThresh, cmap="gray")
        axs[0][2].set_title(
            f"Robust HSV | time:{round(HTime,3)}s | acc:{round(Hacc,3)}")

        axs[1][0].imshow(YThresh, cmap="gray")
        axs[1][0].set_title(
            f"YCgCr | time:{round(YTime,3)}s | acc:{round(Yacc,3)}")

        axs[1][1].imshow(KThresh, cmap="gray")
        axs[1][1].set_title(
            f"k-means | time:{round(KTime,3)}s | acc:{round(Kacc,3)}")
        plt.show() """


if __name__ == "__main__":
    image_path = '/home/frexg/Downloads/lara2018-master/segmentation/dataset/images/test'
    # image_path = '/home/frexg/Documents/Artificial Intelligence Center/BROCOLE/Cropped_dataset/test_data/a'

    c = CoffeeNet()
    if os.path.exists(image_path):
        for dirpath, dirname, filenames in os.walk(image_path):
            for ImageFile in filenames:
                c.classifyImage(image_path, ImageFile)
    else:
        print("Incorrect path")
