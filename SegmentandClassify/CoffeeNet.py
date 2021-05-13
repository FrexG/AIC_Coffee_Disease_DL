from Classifier.ImageSegment import ImageSegment
from Classifier.ycgcr import YCGCR
import os


class CoffeeNet:
    # Path to saved model
    model_path = '/home/frexg/Keras_Practice/models/CoffeeNet_V2_MobileNet'

    def getImagePath(self, image_path, filename):
        # get path of image file
        return os.path.join(image_path, filename)

    def getModelPath(self):
        # get path of model
        return self.model_path

    def classifyImage(self, imagepath, filename):
        # Start Image Segmentation
        # print("IoU HSV")q
        ImageSegment(self.getImagePath(
            imagepath, filename), self.getModelPath(), filename)
        #print("IoU YCgCr")
        #YCGCR(filename, image=self.getImagePath(image_path, filename))


if __name__ == "__main__":
    image_path = '/home/frexg/Downloads/lara2018-master/segmentation/dataset/images/test'
    c = CoffeeNet()
    if os.path.exists(image_path):
        for dirpath, dirname, filenames in os.walk(image_path):
            for ImageFile in filenames:
                c.classifyImage(image_path, ImageFile)
    else:
        print("Incorrent path")
