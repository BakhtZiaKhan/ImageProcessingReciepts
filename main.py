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
        cv2.imshow("Denoised Image", denoised_image)
        
        # Apply sharpening
        print("Applying sharpening...")
        sharpened_image = cv2.filter2D(denoised_image, -1, self.kernels[0])
        cv2.imshow("Sharpened Image", sharpened_image)

        # Apply thresholding methods
        print("Applying thresholding methods...")
        # Global Thresholding
        _, global_thresh = cv2.threshold(sharpened_image, 127, 255, cv2.THRESH_BINARY)
        global_thresh = cv2.bitwise_not(global_thresh)  # Invert colors for OCR
        cv2.imshow("Global Thresholding", global_thresh)

        # Adaptive Mean Thresholding
        adaptive_mean_thresh = cv2.adaptiveThreshold(
            sharpened_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
        )
        adaptive_mean_thresh = cv2.bitwise_not(adaptive_mean_thresh)  # Invert colors for OCR
        cv2.imshow("Adaptive Mean Thresholding", adaptive_mean_thresh)

        # Adaptive Gaussian Thresholding
        adaptive_gaussian_thresh = cv2.adaptiveThreshold(
            sharpened_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )
        adaptive_gaussian_thresh = cv2.bitwise_not(adaptive_gaussian_thresh)  # Invert colors for OCR
        cv2.imshow("Adaptive Gaussian Thresholding", adaptive_gaussian_thresh)

        # Run OCR on thresholded images
        print("Running OCR on thresholded images...")
        config = '--psm 6'  # OCR configuration for single block text
        text_global = pytesseract.image_to_string(global_thresh, config=config)
        text_adaptive_mean = pytesseract.image_to_string(adaptive_mean_thresh, config=config)
        text_adaptive_gaussian = pytesseract.image_to_string(adaptive_gaussian_thresh, config=config)

        # Print OCR results for each method
        print("\nOCR Output for Global Thresholding:")
        print(text_global)

        print("\nOCR Output for Adaptive Mean Thresholding:")
        print(text_adaptive_mean)

        print("\nOCR Output for Adaptive Gaussian Thresholding:")
        print(text_adaptive_gaussian)

        # Calculate word counts for each method
        word_count_global = len(text_global.split())
        word_count_adaptive_mean = len(text_adaptive_mean.split())
        word_count_adaptive_gaussian = len(text_adaptive_gaussian.split())

        print("\nWord Counts:")
        print(f"Global Thresholding Word Count: {word_count_global}")
        print(f"Adaptive Mean Thresholding Word Count: {word_count_adaptive_mean}")
        print(f"Adaptive Gaussian Thresholding Word Count: {word_count_adaptive_gaussian}")

        # Determine the best thresholding method
        word_counts = {
            "Global Thresholding": (word_count_global, global_thresh),
            "Adaptive Mean Thresholding": (word_count_adaptive_mean, adaptive_mean_thresh),
            "Adaptive Gaussian Thresholding": (word_count_adaptive_gaussian, adaptive_gaussian_thresh),
        }
        best_method, (best_word_count, best_thresh_image) = max(word_counts.items(), key=lambda x: x[1][0])
        print(f"\nBest thresholding method: {best_method} with word count {best_word_count}")

        # Display the best thresholded image
        cv2.imshow(f"Best Thresholded Image: {best_method}", best_thresh_image)

        # Wait for user input and close windows
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





