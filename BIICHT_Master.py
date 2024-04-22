import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename 
from matplotlib import image as img
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import datetime
import pandas as pd
from scipy.ndimage import gaussian_filter
import logging 
import sys 
import copy 
from skimage import img_as_ubyte
from tifffile import imwrite

def BIICHT_Master(): 
    # Courtesy of https://patorjk.com/software/taag/#p=testall&f=Graffiti&t=BIICHT
     print("""\n\n\n         
               ██████╗ ██╗██╗ ██████╗██╗  ██╗████████╗
               ██╔══██╗██║██║██╔════╝██║  ██║╚══██╔══╝
               ██████╔╝██║██║██║     ███████║   ██║   
               ██╔══██╗██║██║██║     ██╔══██║   ██║   
               ██████╔╝██║██║╚██████╗██║  ██║   ██║   
               ╚═════╝ ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝  Batch Image Intensity Calculation Helper Tool \n""")

     # Provide initial instructions to the new user 
     print('Welcome to BIICHT (pronounced "beach tea"), your solution for batch microscopy image thresholding and normalization!\n') 
     print('Please select the CSV containing the paths and instructions for the data you would like to process. Your CSV columns should be formatted as follows:\n') 
     print("""
          1   "Data": Each row gives the full filepath to a three-color-channel TIFF image from which pixels will be thresholded,
          summed, and normalized to the associated reference mask defined in the "Mask" column. Image (and associated mask) filepaths may 
          be repeated across rows to enable different processing inputs in the last three columns ("Data_Threshold", "Mask_Threshold", and "Mask_Blur"). 
               Example (from a Windows system): C:\\Users\\kswanberg\\DAPI_lectin_collagen\\GLU01-1_647.tif
           
          2   "Mask": Each row gives the full filepath to a three-color-channel TIFF image of the same dimensionality as the corresponding image 
          referenced in the "Data" column. In most biological research applications this mask is the same tissue slice as the image to be analyzed, 
          such that no registration is needed for reasonable overlap. This mask will be the basis for the masking and normalization of thresholded 
          and summed pixels from the associated data image defined in the "Data" column. Mask (and associated image) filepaths may be repeated 
          across rows to enable different processing inputs in the last three columns ("Data_Threshold", "Mask_Threshold", and "Mask_Blur"). 
               Example (from a Windows system): C:\\Users\\kswanberg\\DAPI_lectin_collagen\\GLU01-1_DAPI.tif
           
          3    "Data_Threshold": Absolute intensity threshold below which "Data" image pixels will not be counted in the final normalized intensity calculation. 
               Example: 100
           
          4    "Mask_Threshold": Absolute intensity threshold below which "Mask" reference image pixels will not be counted in the binarized mask 
          following the blur step before binarization. 
               Example: 20
           
          5    "Mask_Blur": Sigma value defining the extent of Gaussian blur (implemented by scipy.ndimage's "gaussian_filter" function) applied to the "Mask" 
          reference image for image denoising prior to thresholding and binarization. Note that no blurring is applied to any final image; rather, this is a denoising step 
          for defining the binarized mask.  
               Example: 16
           
          Header and full first line of example input CSV: 
           
          Data, Mask, Data_Threshold, Mask_Threshold, Mask_Blur
          C:\\Users\\kswanberg\\DAPI_lectin_collagen\\GLU01-1_647.tif, C:\\Users\\kswanberg\\DAPI_lectin_collagen\\GLU01-1_DAPI.tif, 100, 20, 16
          \n\n""")
     
     # Set working directory to location of file 
     os.chdir(sys.path[0])

     # Create output directory 
     time_for_dirname = datetime.datetime.now() 
     root_dirname = str('BIICHT_Outputs_' + str(time_for_dirname)).replace(' ', '_')
     root_dirname = root_dirname.replace(':', '')
     root_dirname = root_dirname.replace('.', '')
     os.mkdir(root_dirname)

     # Create error log
     log_name = str(root_dirname + '\\' 'BIICHT_log.txt')
     logging.basicConfig(filename=log_name, level=logging.INFO)

     # Adapted from https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
     Tk().withdraw()
     FilePathtoCSV = askopenfilename()
     print(FilePathtoCSV)

     # Import CSV with filepaths to two image types, each in one column 
     try:
          data = pd.read_csv(FilePathtoCSV, sep=None, usecols = ['Data','Mask','Data_Threshold', 'Mask_Threshold', 'Mask_Blur'])
          print("Read input CSV ", FilePathtoCSV)
          logging.info("Read input CSV %s", FilePathtoCSV)
     except: 
          logging.error("Could not read input CSV %s", FilePathtoCSV)

     # Read information in CSV to parse filepaths for images to analyze
     data.head()

     # Determine number of cases to analyze
     num_cases = data.shape[0]

     # Set up an array for analysis output 
     analysis_outputs = []

     # Loop through number of cases and apply masking and averaging analysis to each 
     #for ii in range(num_cases): 
     for ii in range(num_cases): 
          # Define current image data 
          image_loc = data['Data'][ii]
          mask_loc = data['Mask'][ii]
          image_threshold = data['Data_Threshold'][ii]
          mask_threshold = data['Mask_Threshold'][ii]
          mask_blur = data['Mask_Blur'][ii]

          # Run analysis kernel on image and mask pair 
          mask_px, image_px, image_px_normalized,  image_sum_of_intensities_val, image_sum_of_intensities_normalized_val = Masking_to_Average_kernel(image_loc, mask_loc, image_threshold, mask_threshold, mask_blur, root_dirname)
          analysis_outputs.append([image_loc, mask_loc, image_threshold, mask_threshold, mask_blur, int(mask_px), int(image_px), float(image_px_normalized), float(image_sum_of_intensities_val), float(image_sum_of_intensities_normalized_val)])
          plt.close()
     
     analysis_outputs_df = pd.DataFrame(analysis_outputs, columns=['Data', 'Mask', 'Data_Threshold', 'Mask_Threshold', 'Mask_Blur', 'Binary_Mask_Px_N', 'Thresholded_Masked_Image_Px_N', 'Thresholded_Masked_Image_Px_N_Normalized', 'Summed_Intensities', 'Summed_Intensities_Normalized'])

     # Save data outputs to CSV 
     csv_to_save_eu = str(root_dirname + '\\' 'Contrast_statistics_normalized_eu.csv')
     csv_to_save = str(root_dirname + '\\' 'Contrast_statistics_normalized.csv')

     try: 
          analysis_outputs_df.to_csv(csv_to_save, sep=',', index=False) # Note: For European settings of Excel
          analysis_outputs_df.to_csv(csv_to_save_eu, sep=';', index=False) # Note: For European settings of Excel
          print("Saved ", csv_to_save, " and ", csv_to_save_eu)
          logging.info("Saved %s and %s", csv_to_save, csv_to_save_eu)
     except: 
          logging.error("Unable to save %s and %s", csv_to_save, csv_to_save_eu)

def Masking_to_Average_kernel(image_loc_var, mask_loc_var, data_threshold_var, mask_threshold_var, mask_blur_var, root_dirname):
     # Adapted from https://datacarpentry.org/image-processing/07-thresholding.html

     # Load image 
     try:
          image_pre_threshold = cv2.imread(image_loc_var, cv2.IMREAD_UNCHANGED)
          logging.info("Loaded image %s", image_loc_var)
     except:
          logging.error("Could not load image %s", image_loc_var)
     #plt.imshow(image_pre_threshold)
     #plt.show()

     # Define threshold for image data and mask by threshold 
     image_file_title_old = image_loc_var.split('\\')[-1]
     image_file_title = image_file_title_old.replace('.', '_')
     image_file_title = image_file_title.replace('-', '_')
     image_file_title = image_file_title.replace(' ', '_')

     #image_threshold = 1
     image_threshold = data_threshold_var
     try: 
          image_thresholded = Threshold_and_Mask_Image(image_pre_threshold, image_file_title, image_threshold, 'H') 
          print("Thresholded image ", image_loc_var)
          logging.info("Thresholded image %s", image_loc_var)
     except: 
          logging.error("Could not threshold image %s", image_loc_var)

     # Load mask 
     try: 
          mask_pre_threshold_unblurred = cv2.imread(mask_loc_var, cv2.IMREAD_UNCHANGED)
          print("Loaded mask ", mask_loc_var)
          logging.info("Loaded mask %s", mask_loc_var)
     except: 
          logging.error("Could not load mask %s", mask_loc_var)

     # Blur mask 
     try: 
          print("Denoising step 1 (blurring)... ") 
          mask_pre_threshold = gaussian_filter(mask_pre_threshold_unblurred, sigma=mask_blur_var)
          print("Blurred mask ", mask_loc_var)
          logging.info("Blurred mask %s", mask_loc_var)
     except: 
          logging.error("Could not blur mask %s", mask_loc_var)
     #plt.imshow(mask_pre_threshold)
     #plt.show()

     # Define threshold for mask data and mask by threshold 
     mask_file_title_old = mask_loc_var.split('\\')[-1]
     mask_file_title = mask_file_title_old.replace('.', '_')
     mask_file_title = mask_file_title.replace('-', '_')
     mask_file_title = mask_file_title.replace(' ', '_')
     mask_threshold = mask_threshold_var

     # Threshold the denoised mask 
     try: 
          print("Denoising step 2 (thresholding)... ") 
          mask_thresholded = Threshold_and_Mask_Image(mask_pre_threshold, mask_file_title, mask_threshold, 'H')
          print("Thresholded mask ", mask_loc_var) 
          logging.info("Thresholded mask %s", mask_loc_var) 
     except: 
          logging.error("Could not threshold mask %s", mask_loc_var) 

     dir_name = str(root_dirname + '\\' + image_file_title + '_it_' + str(image_threshold) + '_mt_' + str(mask_threshold) + '_sigma_' + str(mask_blur_var))
     os.mkdir(dir_name)

     # Display thresholded image and mask 
     fig = plt.figure(figsize=(20, 24)) 

     fig.add_subplot(2, 3, 1) 
     plt.imshow(mask_pre_threshold_unblurred)
     plt.title('Input Mask')
     fig.add_subplot(2, 3, 2) 
     plt.imshow(mask_pre_threshold)
     plt.title('Blurred Input Mask')
     fig.add_subplot(2, 3, 3) 
     plt.imshow(mask_thresholded)
     plt.title('Blurred Input Mask After Threshold')
     fig.add_subplot(2, 3, 4) 
     plt.imshow(image_pre_threshold)
     plt.title('Input Image')
     fig.add_subplot(2, 3, 5) 
     plt.imshow(image_thresholded)
     plt.title('Input Image After Threshold')

     # Save Plot 1 to file 
     fig_name = str(dir_name + '\\Figure_1_Inputs_Thresholded_Outputs.png')
     try: 
          plt.savefig(fig_name) 
          plt.close()
          logging.info("Figure 1 saved as %s", fig_name) 
     except: 
          logging.error("Figure 1 could not be saved as %s", fig_name) 

     # Apply thresholded mask to thresholded image 
     image_thresholded_and_masked = image_thresholded.copy()
     
     # Convert the thresholded mask to grayscale and then binary 
     try: 
          print("Converting mask to grayscale for binarization... ") 
          #mask_thresholded_grayscale = cv2.cvtColor(mask_thresholded, cv2.COLOR_BGR2GRAY)
          mask_thresholded_grayscale = mask_thresholded.copy(); 
          print("Converted to greyscale for mask ", mask_loc_var) 
          logging.info("Converted to greyscale for mask %s", mask_loc_var) 
     except: 
          logging.error("Could not convert to greyscale for mask %s", mask_loc_var) 
     
     try: 
          ret, mask_thresholded_grayscale_binary_mask = cv2.threshold(mask_thresholded_grayscale, mask_threshold, 255, cv2.THRESH_BINARY)
          print("Converted to binary for mask ", mask_loc_var) 
          logging.info("Converted to binary for mask %s", mask_loc_var) 
     except: 
          logging.info("Could not convert to binary for mask %s", mask_loc_var) 
     
     try: 
          image_thresholded_and_masked_for_analysis = cv2.bitwise_and(image_thresholded_and_masked, image_thresholded_and_masked, mask = img_as_ubyte(mask_thresholded_grayscale_binary_mask)); 
          print("Applied mask ", mask_loc_var, "to image ", image_loc_var) 
          logging.info("Applied mask %s to image %s", mask_loc_var, image_loc_var) 
     except: 
          logging.error("Could not apply mask %s to image %s", mask_loc_var, image_loc_var) 

     try:      
          print("Converting image to grayscale for final intensity calculation... ") 
          #image_thresholded_and_masked_grayscale = cv2.cvtColor(image_thresholded_and_masked, cv2.COLOR_BGR2GRAY)
          image_thresholded_and_masked_grayscale = image_thresholded_and_masked_for_analysis.copy(); 
          print("Converted to greyscale for masked image ", image_loc_var) 
          logging.info("Converted to greyscale for masked image %s", image_loc_var) 
     except: 
          logging.error("Could not convert to greyscale for masked image %s", image_loc_var) 
     #plt.imshow(image_thresholded_and_masked)
     #plt.show() 
     
     fig = plt.figure(figsize=(18, 9)) 

     fig.add_subplot(1, 3, 1) 
     plt.imshow(mask_thresholded_grayscale_binary_mask)
     plt.title('Thresholded Mask as Binary')
     fig.add_subplot(1, 3, 2) 
     plt.imshow(image_thresholded_and_masked_for_analysis)
     plt.title('Thresholded and Masked Image')
     fig.add_subplot(1, 3, 3) 
     plt.imshow(image_thresholded_and_masked_for_analysis)
     plt.imshow(mask_thresholded_grayscale_binary_mask, cmap='jet', alpha=0.5)
     plt.title('Image Result with Mask Overlay')

     fig_name = str(dir_name + '\\Figure_2_Thresholded_Binary_Mask_and_Final_Result.png')
     try: 
          plt.savefig(fig_name)
          plt.close() 
          print("Figure 2 saved as ", fig_name) 
          logging.info("Figure 2 saved as %s", fig_name) 
     except: 
          logging.error("Figure 2 could not be saved as %s", fig_name) 

     # Save final thresholded mask and final thresholded and masked image as TIFFs
     img1_name = str(dir_name + '\\00_Thresholded_Binary_Mask.tif'); 
     try: 
          imwrite(img1_name, img_as_ubyte(mask_thresholded_grayscale_binary_mask), imagej=True); 
          logging.info("Image 1 saved as %s", img1_name); 
     except: 
          logging.error("Image 1 could not be saved as %s", img1_name); 
     
     img2_name = str(dir_name + '\\01_Masked_and_Thresholded_Image.tif'); 
     try: 
          imwrite(img2_name, image_thresholded_and_masked_for_analysis, imagej=True); 
          logging.info("Image 2 saved as %s", img2_name); 
     except:
          logging.error("Image 2 could not be saved as %s", img2_name); 

     # Average the image pixel intensities normalized by mask area 
     mask_number_of_px = mask_thresholded_grayscale_binary_mask[mask_thresholded_grayscale_binary_mask != 0].size # Calculates number of pixels to be considered 
     image_sum_of_intensities = np.sum(image_thresholded_and_masked_for_analysis, dtype=np.int64) # Calculates sum of pixel intensities
     #image_sum_of_intensities = np.sum(image_thresholded_and_masked_grayscale); 
     image_thresholded_and_masked_number_of_px = image_thresholded_and_masked_for_analysis[image_thresholded_and_masked_for_analysis != 0].size # Note that this has to be converted to greyscale first 
     image_thresholded_and_masked_number_of_px_normalized = image_thresholded_and_masked_number_of_px / mask_number_of_px; 
     image_sum_of_intensities_normalized = image_sum_of_intensities / mask_number_of_px; 

     try: 
          print("BIICHT processing completed for ", image_loc_var) 
          logging.info("BIICHT processing completed for %s", image_loc_var) 
          return [mask_number_of_px, image_thresholded_and_masked_number_of_px, image_thresholded_and_masked_number_of_px_normalized, image_sum_of_intensities, image_sum_of_intensities_normalized]
     except: 
          logging.error("BIICHT exited with an error for ", image_loc_var) 

def Threshold_and_Mask_Image(image_to_threshold_var, image_name_var, threshold_value_var, polarity_var): 
     
      # Convert image to grayscale as threshold preprocessing 
     #image_pre_threshold_grayscale = cv2.cvtColor(image_pre_threshold_grayscale, cv2.COLOR_BGR2GRAY)
     image_pre_threshold_grayscale = image_to_threshold_var.copy(); 
     histogram, bin_edges = np.histogram(image_pre_threshold_grayscale, bins = 512)
     fig, ax = plt.subplots()
     plt.plot(histogram)

     #plt.title('Grayscale Histogram for '+ image_name_var)
     #plt.show()

     # Define threshold for binary mask 
     if polarity_var == 'H':
          ret, image_threshold_binary_mask = cv2.threshold(image_pre_threshold_grayscale, threshold_value_var, 255, cv2.THRESH_BINARY)
     
     else:
          ret, image_threshold_binary_mask = cv2.threshold(image_pre_threshold_grayscale, threshold_value_var, 255, cv2.THRESH_BINARY_INV)

     #plt.imshow(image_threshold_binary_mask, cmap='gray')
     #plt.show() 

     # Use threshold mask to select only part of the image 
     image_thresholded_var = image_to_threshold_var.copy()
     image_thresholded_var = cv2.bitwise_and(image_thresholded_var, image_thresholded_var, mask = img_as_ubyte(image_threshold_binary_mask))
     #plt.imshow(image_thresholded_var)
     #plt.show() 

     return image_thresholded_var  

BIICHT_Master()






