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
        self.kernels = [
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        ]

    def enhanceDarkImage(self, image):
        ImageGamma = 1.5

        newGamma = np.power(image / 255.0, ImageGamma)
        newGamma = (newGamma * 255).astype(np.uint8)

        # yuv of new gamma image
        yuvImage = cv2.cvtColor(newGamma, cv2.COLOR_BGR2YUV)

        # equalise the intensity which is the Y channel
        yuvImage[ :, :, 0] = cv2.equalizeHist(yuvImage[ :, :, 0])

        # convert image back to BGR colour space
        enchancedImage = cv2.cvtColor(yuvImage, cv2.COLOR_YUV2BGR)

        return enchancedImage

    def enhanceBrightImage(self, image):
        # aplha controls contrast and beta controls brightness
        alpha = 1.5
        beta = -10

        # adjust the contrast and brightness of the image
        adjustedImage = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # yuv of new gamma image
        yuvImage = cv2.cvtColor(adjustedImage, cv2.COLOR_BGR2YUV)

        # equalise the intensity which is the Y channel
        yuvImage[:, :, 0] = cv2.equalizeHist(yuvImage[:, :, 0])

        # convert image back to BGR colour space
        enchancedImage = cv2.cvtColor(yuvImage, cv2.COLOR_YUV2BGR)

        return enchancedImage


    def checkBestbrightnesss(self, image):
        pass

    def ocr_word_count(self, image):
        # first use the orintation function to rotate the image to the best view vertical
        # then use the output from that image to count how many word are visible

        correctOrientation = self.orientation(image, self.rgbImage)

        # then use pytesseract to count the words on the image
        text = pytesseract.image_to_string(correctOrientation)
        words = text.split()
        # need to return the orientated image also
        return len(words)

    def checkWordCount(self, image):
        # preform OCr on the original image for a base line
        original_word_count = self.ocr_word_count(image)
        print(f"Original Word Count: {original_word_count}")

        # use different shparing kernels for best result
        best_image = image  # Start with the original
        best_word_count = original_word_count

        for kernel in self.kernels:
            # Apply sharpening kernel to the image
            sharpened = cv2.filter2D(image, -1, kernel)

            # Use ocr on the image to see if the sharpeing has any impact
            sharpened_word_count = self.ocr_word_count(sharpened)
            print(f"{kernel} Sharpened word count: {sharpened_word_count}")

            # see if the sharpeing kernel has any impact
            if sharpened_word_count > best_word_count:
                best_word_count = sharpened_word_count
                best_image = sharpened  # Update best image



        # print(f"Best kernel: {best_image}")

        return best_image

    # def sharepenImage(self, removedNoiseImage):
    #     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #
    #     sharpened_image = cv2.filter2D(removedNoiseImage, -1, kernel)
    #
    #     return sharpened_image

    def removingNoise(self, image):
        filtered_image = cv2.fastNlMeansDenoising(image, None, 7, 7, 21)

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

    # def exampleusage(self):
    #     correctOrientation = self.orientation(self.image, self.rgbImage)
    #
    #     cv2.imshow("CorrectedOrientation", correctOrientation)
    #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # def exampleusage(self):
    #     removeNoise = self.removingNoise(self.image)
    #
    #     cv2.imshow("noise removal", removeNoise)
    #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def exampleusage(self):
        shaprness = self.checkWordCount(self.image)

        cv2.imshow("Sharpness best image", shaprness)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def exampleusage(self):
    #     orient = self.orientation(self.image, self.rgbImage)
    #
    #     cv2.imshow("Best oreint", orient)
    #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


"""Leave all below for example usage"""

reciept = ProcessReciept("")
reciept.exampleusage()





