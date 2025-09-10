#This code was added to the end of each modular section as it was built in order to unit test each one within the terminal
"""
if __name__ == "__main__":
    test_filename = "T0004-04-06-2019_BL (1).JPG"
    raw_folder = "data/raw_images"  # Match parameter name exactly

    img, filename = gaussian_denoise(test_filename, raw_folder=raw_folder) #change function name as needed

    # Display the image
    cv2.imshow("Gaussian Results", img) #change results names as needed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""