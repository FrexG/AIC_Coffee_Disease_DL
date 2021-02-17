from Classifier.ImageSegment import ImageSegment


class CoffeeNet:
    # Path to saved model
    model_path = '/home/frexg/Downloads/CoffeeNet_V2-20201228T083108Z-001/CoffeeNet_V2'
    image_path = '/home/frexg/Keras_Practice/img/8.jpg'

    def getImagePath(self):
        # get path of image file
        return self.image_path

    def getModelPath(self):
        # get path of model
        return self.model_path

    def classifyImage(self):
        # Start Image Segmentation
        ImageSegment(self.getImagePath(), self.getModelPath())


if __name__ == "__main__":
    c = CoffeeNet()
    c.classifyImage()
