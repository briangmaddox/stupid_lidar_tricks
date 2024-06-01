import cv2
import numpy as np
import argparse
import sys
import config

# *************************************************************************************************
def saliency_factory(in_type: str):
    saliency = None

    if in_type == "spectralresidual":
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    elif in_type == "finegrained":
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    elif in_type == "objectnessbing":
        saliency = cv2.saliency.ObjectnessBING_create()
        saliency.setTrainingPath(config.BING_MODEL_PATH)

    return saliency

# *************************************************************************************************
def main():
    # Process our command line arguments
    ap = argparse.ArgumentParser(description="OpenCV Saliency test program.")
    ap.add_argument("-i", 
                    "--input", 
                    type=str, 
                    required=True, 
                    help="Input image to process.")
    ap.add_argument("-g", 
                    "--geotiff", 
                    type=bool, 
                    required=False, 
                    help="Whether or not to process a GeoTIFF.")
    args = vars(ap.parse_args())

    is_geotiff = False
    input_image = None
    success = False
    saliency_map = None

    static_saliency = ["spectralresidual", "finegrained"]

    if args["geotiff"]:
        is_geotiff = True

    if is_geotiff:
        # Load a 32-bit grayscale TIFF image
        input_image = cv2.imread(args["input"],
                                 cv2.IMREAD_UNCHANGED)
        # Normalize the image to the 0-255 range
        normalized_image = cv2.normalize(input_image,
                                         None,
                                         alpha=0,
                                         beta=255,
                                         norm_type=cv2.NORM_MINMAX)

        # Convert to 8-bit
        image_8bit = np.uint8(normalized_image)

        # Convert single channel image to 3-channel (BGR) image
        input_image = cv2.cvtColor(image_8bit,
                                   cv2.COLOR_GRAY2BGR)
    else:
        input_image = cv2.imread(args["input"])

    # Check if the image is loaded properly
    if input_image is None:
        print("Could not open or find the image")
        sys.exit()

    # Instantiate our saliency object
    saliency = saliency_factory(config.OPENCV_SALIENCY)

    # Now based on the type, compute the saliency map
    if config.OPENCV_SALIENCY in static_saliency:
        # Run the saliency detector
        (success, saliency_map) = saliency.computeSaliency(input_image)

        if success:
            # Convert the saliency map to an 8-bit image for visualization
            saliency_map = (saliency_map * 255).astype("uint8")

            # Now display it and the original images
            cv2.imshow("Original", input_image)
            cv2.imshow("Saliency", saliency_map)
            cv2.waitKey(0)
        else:
            print("An error happened while running saliency!")
    elif config.OPENCV_SALIENCY == "objectnessbing":
        # Run the saliency detector
        (success, saliency_map) = saliency.computeSaliency(input_image)

        # Get the number of detections
        saliency_detections = saliency_map.shape[0]

        # Copy the input image
        image_copy = input_image.copy()

        # Now display the detections, limiting to the config.NUMBER_DETECTIONS value, making sure we 
        # take the min of the actual number of detections vs how many we want to show.
        for detection in range(0, min(saliency_detections, config.NUMBER_DETECTIONS)):
           	# extract the bounding box coordinates
            (start_x, start_y, end_x, end_y) = saliency_map[detection].flatten()

            # randomly generate a color for the object and draw it on the image
            color = np.random.randint(0, 255, size=(3,))
            color = [int(c) for c in color]
            cv2.rectangle(image_copy, (start_x, start_y), (end_x, end_y), color, 2)

        # show the output image
        cv2.imshow("Image", image_copy)
        cv2.waitKey(0)

    # do a bit of cleanup
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
