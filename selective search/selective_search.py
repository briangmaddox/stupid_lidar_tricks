import cv2
import numpy as np
import argparse
import sys

# *********************************************************************************************************************
def add_selective_search_strategies(in_selective_search):
    """This functions adds various strategies to the input selective search object

    Args:
        in_selective_search (cv2.ximgproc_segmentation): Input object to add strategies to

    Returns:
        cv2.ximgproc_segmentation: Object with the added strategies
    """

    # Add in our strategies
    gs = cv2.ximgproc.segmentation.createGraphSegmentation()
    in_selective_search.addGraphSegmentation(gs)

    strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    strategy_fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
    strategy_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
    strategy_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
    strategy_multiple = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(strategy_color,
                                                                                                    strategy_fill,
                                                                                                    strategy_size,
                                                                                                    strategy_texture)
    in_selective_search.addStrategy(strategy_multiple)

    return in_selective_search


# *********************************************************************************************************************
def main():
    # Process our command line arguments
    ap = argparse.ArgumentParser(description="Selective Search test program.")
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

    if args["geotiff"]:
        is_geotiff = True

    if is_geotiff:
        # Load a 32-bit grayscale TIFF image
        input_image = cv2.imread(args["input"],
                                 cv2.IMREAD_UNCHANGED)
    else:
        input_image = cv2.imread(args["input"])

    # Check if the image is loaded properly
    if input_image is None:
        print("Could not open or find the image")
        sys.exit()
    
    if is_geotiff:
        # Normalize the image to the 0-255 range
        normalized_image = cv2.normalize(input_image,
                                         None,
                                         alpha=0,
                                         beta=255,
                                         norm_type=cv2.NORM_MINMAX)

        # Convert to 8-bit
        image_8bit = np.uint8(normalized_image)

        # Convert single channel image to 3-channel (BGR) image
        image_color = cv2.cvtColor(image_8bit,
                                   cv2.COLOR_GRAY2BGR)
    else:
        image_color = input_image

    # Create a selective search segmentation object
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # Set the input image on which we will run the segmentation
    selective_search.setBaseImage(image_color)

    # Switch to quality mode
    selective_search.switchToSelectiveSearchQuality()
    
    # Add in the select search strategies to try to see if it performs netter.
    selective_search = add_selective_search_strategies(selective_search)

    # Run selective search segmentation
    regions_of_interest = selective_search.process()

    # Number of region proposals to show
    number_of_rectangles = 20

    # Create a copy of original image
    output_image = image_color.copy()

    # Draw rectangles on the image
    for i, rect in enumerate(regions_of_interest):
        if i < number_of_rectangles:
            x, y, w, h = rect
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

    # Show output
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
