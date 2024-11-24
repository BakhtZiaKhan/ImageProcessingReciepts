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

    def imageThresholdingTextExtraction(self):
        # Apply noise removal
        print("Applying noise removal...")
        denoised_image = self.removingNoise(self.greyImage)

        #Apply sharpening
        print("Applying sharpening...")
        sharpened_image = cv2.filter2D(denoised_image, -1, self.kernels[0])  # Using the first kernel

        #Apply global thresholding
        print("Applying global thresholding...")
        _, th1 = cv2.threshold(sharpened_image, 127, 255, cv2.THRESH_BINARY)
        th1 = cv2.bitwise_not(th1)  # Invert colors

        #Apply adaptive mean thresholding
        print("Applying adaptive mean thresholding...")
        th2 = cv2.adaptiveThreshold(
            sharpened_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        th2 = cv2.bitwise_not(th2)  # Invert colors

        #Apply adaptive Gaussian thresholding
        print("Applying adaptive Gaussian thresholding...")
        th3 = cv2.adaptiveThreshold(
            sharpened_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        th3 = cv2.bitwise_not(th3)  # Invert colors

        #Evaluate word count for each thresholded image
        print("Evaluating word counts for thresholded images...")
        word_count_th1 = self.ocr_word_count(th1)
        word_count_th2 = self.ocr_word_count(th2)
        word_count_th3 = self.ocr_word_count(th3)

        print(f"Global Thresholding Word Count: {word_count_th1}")
        print(f"Adaptive Mean Thresholding Word Count: {word_count_th2}")
        print(f"Adaptive Gaussian Thresholding Word Count: {word_count_th3}")

        #Select the best thresholding method based on word count
        word_counts = {
            "Global Thresholding": (word_count_th1, th1),
            "Adaptive Mean Thresholding": (word_count_th2, th2),
            "Adaptive Gaussian Thresholding": (word_count_th3, th3),
        }
        best_method, (best_word_count, best_thresh_image) = max(word_counts.items(), key=lambda x: x[1][0])
        print(f"Best thresholding method: {best_method} with word count {best_word_count}")

        #Display all thresholded images
        cv2.imshow("Global Thresholding", th1)
        cv2.imshow("Adaptive Mean Thresholding", th2)
        cv2.imshow("Adaptive Gaussian Thresholding", th3)

        #Display the best thresholded image
        cv2.imshow(f"Best Thresholded Image: {best_method}", best_thresh_image)

        #Create a rectangular kernel for dilation
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # Apply dilation
        print("Applying dilation...")
        dilation = cv2.dilate(best_thresh_image, rect_kernel, iterations=1)

        #Find contours on the dilated image
        print("Finding contours...")
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #Create a copy of the original image for drawing rectangles
        im2 = self.image.copy()

        #Prepare a text file to store extracted text
        print("Extracting text...")
        with open("recognized.txt", "w") as file:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # Draw rectangles on the image
                cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the text region
                cropped = self.image[y:y + h, x:x + w]

                # Apply OCR on the cropped image
                text = pytesseract.image_to_string(cropped)

                # Write the extracted text to the file
                file.write(text)
                file.write("\n")

        # Step 14: Display processed image with detected text regions
        cv2.imshow("Detected Text Regions", im2)

        # Wait for key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()





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
        print("Performing image thresholding and text extraction...")
        self.imageThresholdingTextExtraction()

        print("Testing OCR with sharpening kernels...")
        shaprness = self.checkWordCount(self.image)

        # Display the best-sharpened image
        cv2.imshow("Sharpness best image", shaprness)

        # Wait for key press and close all windows
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





