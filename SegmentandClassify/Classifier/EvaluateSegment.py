import cv2 as cv
import numpy as np
import os


class Evaluator:
    test_image_directory = '/home/frexg/Downloads/lara2018-master/segmentation/dataset/train_binary'
    test_image_path = None
    segmented_image = None
    imageName = None
    time_taken = 0
    csv_file = "./evaluation_results.csv"

    def __init__(self, image, imageFileName, time_taken):
        self.segmented_image = image
        self.time_taken = time_taken
        self.imageName = imageFileName.split('.')[0]
        # print(self.imageName)
        self.test_image_path = os.path.join(
            self.test_image_directory, f'{self.imageName}_mask.png')
        self.IntersectionOverUnion()
        return

    def IntersectionOverUnion(self):
        test_image = cv.imread(self.test_image_path)
        test_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
        # print(test_image.shape)

        Intersection = np.logical_and(test_image, self.segmented_image)
        Union = np.logical_or(test_image, self.segmented_image)

        IoU = np.sum(Intersection) / np.sum(Union)

        print(IoU)

        """ f = open(self.csv_file, "a")
        f.write('\n')
        f.write(self.imageName)
        f.write(',')
        f.write(f'{IoU}')
        f.write(',')
        f.write(f'{self.time_taken}')
        f.close()
 """
