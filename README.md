# BIICHT: Batch Image Intensity Calculation Helper Tool  

BIICHT (Batch Image Intensity Calculation Helper Tool) automatically calculates the number of pixels in an image that clear a user-defined threshold, and then normalizes that number to the total pixels in a binary mask derived from a second reference image that has been preprocessed according to a user-defined Gaussian blur + intensity threshold step for mask image denoising prior to binarization. Users may define a range of multiple input parameters per image and mask pair, making this tool useful for systematic investigation of varied analysis approaches. The original use case for BIICHT was normalizing pixel counts beyond a certain fluorescence intensity threshold for wavelength-647 nm emissions from brain slices normalized to DAPI-stained images of the same slices, but this tool can be employed out-of-box for any pair of three-channel TIFF files for which thresholded pixel intensities in one must be normalized to an (optionally denoised) mask derived from the other, as well as further adapted to a wider range of applications.  

### Inputs

Upon function run the user will be prompted to select an input CSV file with predefined column contents and a user-defined number of rows. The input columns and contents follow the below format: 

* Data: Each row gives the full filepath to a three-color-channel TIFF image from which pixels will be thresholded, summed, and normalized to the associated reference mask defined in the "Mask" column. Image (and associated mask) filepaths may be repeated across rows to enable different processing inputs in the last three columns ("Data_Threshold", "Mask_Threshold", and "Mask_Blur"). Example (from a Windows system): C:\Users\Aquamentus\Documents\DAPI_lectin_collagen\GLU01-1_647.tif

* Mask: Each row gives the full filepath to a three-color-channel TIFF image of the same dimensionality as the corresponding image referenced in the "Data" column. In most biological research applications this mask is the same tissue slice as the image to be analyzed, such that no registration is needed for reasonable overlap. This mask will be the basis for the masking and normalization of thresholded and summed pixels from the associated data image defined in the "Data" column. Mask (and associated image) filepaths may be repeated across rows to enable different processing inputs in the last three columns ("Data_Threshold", "Mask_Threshold", and "Mask_Blur"). Example (from a Windows system): C:\Users\Aquamentus\Documents\DAPI_lectin_collagen\GLU01-1_DAPI.tif

* Data_Threshold: Absolute intensity threshold below which "Data" image pixels will not be counted in the final normalized intensity calculation. Example: 100

* Mask_Threshold: Absolute intensity threshold below which "Mask" reference image pixels will not be counted in the binarized mask following the blur step before binarization. Example: 20

* Mask_Blurring: Sigma value defining the extent of Gaussian blur (implemented by scipy.ndimage's "gaussian_filter" function) applied to the "Mask" reference image prior to thresholding and binarization. Example: 16

### Outputs

BIICHT outputs into the working directory (typically the folder in which the script lives unless otherwise defined) a directory with the prefix "BIICHT_Outputs_" and a suffix based on the date-time, and which contains the following contents: 

* One folder per row in the input csv, named after the file input to the "Data" field of that row, that contains two figures: "Figure_1_Inputs_Thresholded_Outputs.png," which shows each step of blurring (according to the "Mask_Blurring" input) and thresholding (according to the "Mask_Threshold" input) (i.e. denoised) for the "Mask" image and thresholding for the "Data" image (according to the "Data_Threshold" input), and "Figure_2_Thresholded_Binary_Mask_and_Final_Result.png," which displays the final denoised mask, the masked and thresholded "Data" image, and an overlay of the two 

* One output csv in the root directory of the folder in which each row contains the input conditions for the analysis in question, with three additional output columns "Binary_Mask_Px_N" for the number of pixels in the final denoised binary mask, "Thresholded_Masked_Image_Px_N" for the number of pixels in the masked and thresholded data image, and "Thresholded_Masked_Image_Px_N_Normalized" for the proportion of the former also occupied by the latter (out of a maximum P=1). Note that this CSV is by default written with ';' as the separator, for European systems, in line 56 of the code. 


### Citation 

Work that employed code from BIICHT can cite it as follows: 

Swanberg, K.M. (2023). BIICHT: Batch Image Intensity Calculation Helper Tool v. 1.0. Source code. https://github.com/kswanberg/BIICHT.


### Developer

Please send comments and questions to [Kelley Swanberg](mailto:kelley.swanberg@med.lu.se). 