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
        pass

    def enhanceBrightImage(self, image):
        pass

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

    def imageThresholdingTextExtraction(self,):
        #Apdative thresholding  
       
       #Adaptive mean Global
        ret,th1 = cv2.threshold(self.greyImage,127,255,cv2.THRESH_BINARY)
        th1 = cv2.bitwise_not(th1) #inverse colours
        
        #Adaptive mean thresh
        th2 = cv2.adaptiveThreshold(self.greyImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        th2 = cv2.bitwise_not(th2) #inverse colours
        
        #Adaptive mean Gaussian
        th3 = cv2.adaptiveThreshold(self.greyImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        th3 = cv2.bitwise_not(th3)#inverse colours
       
        cv2.imshow("Adaptive mean Global", th1)
        cv2.imshow("Adaptive mean Thresh", th2)
        cv2.imshow("Adaptive mean Gaussian", th3)
       
        cv2.waitKey(0)  # Wait for key press to close images
        cv2.destroyAllWindows()

    
       
        # word kernel
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # apply dilation
        dilation = cv2.dilate(th1, rect_kernel, iterations=1)
        dilation = cv2.dilate(th2, rect_kernel, iterations=1)
        dilation = cv2.dilate(th3, rect_kernel, iterations=1)

        # find countors
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        # create copy of image
        im2 =self.image.copy()

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
        self.imageThresholdingTextExtraction()

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

reciept = ProcessReciept(r"C:\Users\bakht\Documents\ImageProcessingReciepts\images\Screenshot 2024-10-23 at 16.46.47.png")
reciept.exampleusage()