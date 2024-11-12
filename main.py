import imutils
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

class ProcessReciept:
    def __init__(self, image):
        self.image = cv2.imread(image)
        self.rgbImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.greyImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def sharepenImage(self, removedNoiseImage):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        sharpened_image = cv2.filter2D(removedNoiseImage, -1, kernel)

        return sharpened_image

    def removingNoise(self, image):
        filtered_image = cv2.GaussianBlur(image, (7, 7), 0)

        return filtered_image

    def enhancingImage(self, sharpenedImage):
        # convert image to hsv
        hsvImage = cv2.cvtColor(sharpenedImage, cv2.COLOR_RGB2HSV)

        # Adjusts the hue by multiplying it by 0.7
        hsvImage[:, :, 0] = hsvImage[:, :, 0] * 0.7
        # Adjusts the saturation by multiplying it by 1.5
        hsvImage[:, :, 1] = hsvImage[:, :, 1] * 1.5
        # Adjusts the value by multiplying it by 0.5
        hsvImage[:, :, 2] = hsvImage[:, :, 2] * 0.5

        return hsvImage

    def imageThresholdingTextExtraction(self, image, greyImage):
        # OSU thresholding
        ret, thresh1 = cv2.threshold(greyImage, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # word kernel
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # apply dilation
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        # find countors
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        # create copy of image
        im2 = image.copy()

        # text file of extracted text
        file = open("recognized.txt", "w+")
        file.write("")
        file.close()

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Drawing a rectangle on copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]

            # Open the file in append mode
            file = open("recognized.txt", "a")

            # Apply OCR on the cropped image
            text = pytesseract.image_to_string(cropped)

            # Appending the text into file
            file.write(text)
            file.write("\n")

            # Close the file
            file.close()

    def orientation(self, image, rgbImgae):
        results = pytesseract.image_to_osd(rgbImgae, output_type=Output.DICT)

        rotatedImage = imutils.rotate_bound(image, angle=results["rotate"])

        return rotatedImage

    def exampleusage(self):
        correctOrientation = self.orientation(self.image, self.rgbImage)

        cv2.imshow("CorrectedOrientation", correctOrientation)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


"""Leave all below for example usage"""

reciept = ProcessReciept("/Users/jonkehoe/PycharmProjects/ImageProcessingCollege/GroupAssignment/reditReciept.jpg")
reciept.exampleusage()