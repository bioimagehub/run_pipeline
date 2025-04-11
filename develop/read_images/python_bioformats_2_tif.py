import imagej
import scyjava

def open_and_save_image(input_path: str, output_path: str):
    """
    Opens an image with Bio-Formats using pyimageJ and saves it as a TIFF file.

    Args:
        input_path (str): The path to the input image file (must be a supported format).
        output_path (str): The path where the output TIFF file will be saved.
    """
    try:
        # Use scyjava to ensure necessary dependencies are available
        scyjava.config_java()
        scyjava.add_maven_repo('https://repo1.maven.org/maven2/')
        scyjava.add_maven_dependency('org.scijava', 'bioformats', '6.9.0')  # Adjust version as necessary
        scyjava.add_maven_dependency('org.scijava', 'scijava-commons', '2.8.0')  # Adjust version as necessary
        scyjava.add_maven_dependency('org.scijava', 'imagej', '2.1.0')  # Adjust version as necessary

        # Initialize ImageJ
        ij = imagej.init('sc.fiji:fiji', mode='interactive')

        # Open image using Bio-Formats
        image = ij.io().open(input_path)
        
        # Save the image as TIFF
        ij.io().save(image, output_path)

        print(f"Image saved successfully as {output_path}")

    except Exception as e:
        print(f"Error while processing the image: {e}")
    finally:
        ij.dispose()  # Clean up and close ImageJ



# Example usage:
open_and_save_image(r"C:\Users\oodegard\Desktop\bfconvert_example\input\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.ims", r"C:\Users\oodegard\Desktop\bfconvert_example\input_tif\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.tif")

