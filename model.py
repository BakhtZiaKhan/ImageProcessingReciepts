import imutils
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import matplotlib.pyplot as plt 


class ProcessReciept:
    def __init__(self, image):
        self.image = cv2.imread(image)
        self.rgbImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.greyImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.kernels = [
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        ]
        self.sharpenedValues = [0]
        self.brightnessValues = [0]
        self.imageContrastValues = [1.0]
        self.reduceNoiseValues = [0]
        self.imageThresholdValues = [True]
        self.adaptiveThresholdBlockSizes = [11, 15, 19]  # Must be odd and >=3 , default 11
        self.adaptiveThresholdConstants = [2, 5, 7] # default 2
        #self.sharpenedValues = [0, 1, 2]
        #self.brightnessValues = [0, 50, 100]
        #self.imageContrastValues = [1.0, 1.5, 2.0]
        #self.reduceNoiseValues = [0, 15, 30]
        #self.imageThresholdValues = [True, False]

    def orientation(self, image):
        try:
            # pytesseract image rotation calculation
            results = pytesseract.image_to_osd(image, output_type=Output.DICT)

            # correction if image has already been rotated
            resultAngle = results.get("rotate", 0)

            # error prevention if rotation angle is already 0
            if resultAngle != 0:
                rotatedImage = imutils.rotate_bound(image, angle=resultAngle)
                return rotatedImage
            return image
        except pytesseract.TesseractError as e:
            # return image if rotation error
            print(f"Rotation failure: {e}")
            return image

    def enhancemntParameters(self, imageEnhancment, OCRWordCount):
        # call orientation on image to ensure valid text OCR
        correctRotation = self.orientation(self.rgbImage)

        enhancedWordCount = 0
        enhancedImage = None
        enhancedParams = {}

        # loop through each parameter for different combinations
        for sharpenedValue in self.sharpenedValues:
            for brightnessValue in self.brightnessValues:
                for imageContrastValue in self.imageContrastValues:
                    for imageThresholdValue in self.imageThresholdValues:
                        for reduceNoiseValue in self.reduceNoiseValues:
                            for blockSize in self.adaptiveThresholdBlockSizes:
                                for C in self.adaptiveThresholdConstants:
                                    # now enhance the image based on each value
                                    imageEnhancement = imageEnhancment(
                                        cv2.cvtColor(correctRotation, cv2.COLOR_BGR2GRAY),
                                        sharpenedValue,
                                        brightnessValue,
                                        imageContrastValue,
                                        reduceNoiseValue,
                                        imageThresholdValue,
                                        blockSize,
                                        C
                                    )
        
                                    # test new parameters for word count
                                    enhancedCount = OCRWordCount(imageEnhancement)
        
                                    # if word count is better then update parameters
                                    if enhancedCount > enhancedWordCount:
                                        enhancedWordCount = enhancedCount
                                        enhancedImage = imageEnhancement
                                        enhancedParams = {
                                            "sharpenedValue": sharpenedValue,
                                            "brightnessValue": brightnessValue,
                                            "imageContrastValue": imageContrastValue,
                                            "reduceNoiseValue": reduceNoiseValue,
                                            "imageThresholdValue": imageThresholdValue,
                                            "blockSize": blockSize,
                                            "C": C
                                        }
        return enhancedImage, enhancedParams, enhancedWordCount

    def adpativeImageEnhancement(self):

        def addImageEnhanements(image, sharpenedValue, brightnessValue, imageContrastValue, reduceNoiseValue, imageThreshold, blockSize, C):

            # make sure that the input image has three colour channels
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # convert to YUV colour space to enhance luminance
            yuvImage = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuvImage)

            # sharpen image based on input to kernel
            if sharpenedValue > 0:
                kernel = np.array([[0, -1, 0], [-1, 5 + sharpenedValue, -1], [0, -1, 0]])
                y = cv2.filter2D(y, -1, kernel)

            # image brightness enhancement
            y = cv2.convertScaleAbs(y, alpha=imageContrastValue, beta=brightnessValue)

            # merge channels back together
            yuvEnhancedImage = cv2.merge([y, u, v])
            enhancedImage = cv2.cvtColor(yuvEnhancedImage, cv2.COLOR_YUV2BGR)

            # image noise reduction
            if reduceNoiseValue > 0:
                enhancedImage = cv2.fastNlMeansDenoising(enhancedImage, None, reduceNoiseValue, 7, 21)

            # TODO Bakht implement image thresholding
            if imageThreshold:
                greyImage = cv2.cvtColor(enhancedImage, cv2.COLOR_BGR2GRAY)
                enhancedImage = cv2.adaptiveThreshold(
                    greyImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C
                )

            return enhancedImage

        def ocr_word_count(image):

            # then use pytesseract to count the words on the image
            text = pytesseract.image_to_string(image, config="--psm 6")
            words = text.split()
            # need to return the orientated image also
            return len(words)

        # call main function to find the best parameter
        enhancedImage, enhancedParams, enhancedWordCount = self.enhancemntParameters(
            addImageEnhanements,
            ocr_word_count
        )

        # testing
        print(f"Best Parameters: {enhancedParams}")
        print(f"Words detected {enhancedWordCount}")
        return enhancedImage,enhancedParams

    #Find total amout: Finds next string containing an integer after a string equal or 1 character away from "total"
    #If no string similar to total is found, the last string containing an integer is returned
    def get_total_amount(self, scanned_strings):
        def is_similar_to_total(word):
            target = 'total'
            word = word.lower()
            m, n = len(target), len(word)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m):
                for j in range(n):
                    if target[i] == word[j]:
                        dp[i+1][j+1] = dp[i][j] + 1
                    else:
                        dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
            lcs_length = dp[m][n]
            return lcs_length >= 4

        # Convert all strings to lowercase for case-insensitive comparison
        lower_strings = [s.lower() for s in scanned_strings]
        try:
            # Find index of word similar to 'total'
            total_index = next(i for i, s in enumerate(lower_strings) if is_similar_to_total(s))
            # Start from the index after 'total'
            for s in scanned_strings[total_index+1:]:
                if any(c.isdigit() for c in s):
                    return s
            # If 'total' is found but no number after it, return "no total found"
            return "no total found"
        except StopIteration:
            #'total' not found, return last string containing a number
            for s in reversed(scanned_strings):
                if any(c.isdigit() for c in s):
                    return s
            #If no numbers found, return "no total found"
            return "no total found"

    def exampleusage(self):
        imageEnhancement, bestParams = self.adpativeImageEnhancement()
        
        text = pytesseract.image_to_string(imageEnhancement, config="--psm 6")
        
        print(f"Detected text: {text}")
        print(f"Best Parameters: {bestParams}")

        scanned_strings = text.split()

        # Call get_total_amount and print the result
        total_amount = self.get_total_amount(scanned_strings)
        print(f"Total Amount Detected: {total_amount}")
        
        #cv2.imshow("Enhanced Image", imageEnhancement)
        #cv2.imshow("Original Image", self.image)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        plt.figure(figsize=(10, 5))

        # Show the original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        # Show the enhanced image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(imageEnhancement, cv2.COLOR_BGR2RGB))
        plt.title("Enhanced Image")
        plt.axis("off")

        # Display the plots
        plt.tight_layout()
        plt.show()


"""Leave all below for example usage"""

reciept = ProcessReciept("\")
reciept.exampleusage()

#for running multiple images, need to correct file names
#image_files = ["image1.png", "image2.png", "image3.png", 
#               "image4.png", "image5.png", "image6.png",
#               "image7.png", "image8.png", "image9.png",
#               "image10.png", "image11.png", "image12.png", 
#               "image13.png"]
#for file_name in image_files:
#    reciept = ProcessReciept(file_name)
#    reciept.exampleusage()