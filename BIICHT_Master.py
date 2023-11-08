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

def BIICHT_Master(): 
    
     # Adapted from https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
     Tk().withdraw()
     FilePathtoCSV = askopenfilename()
     print(FilePathtoCSV)

     # Import CSV with filepaths to two image types, each in one column 
     data = pd.read_csv(FilePathtoCSV, sep=';', usecols = ['Data','Mask','Data_Threshold', 'Mask_Threshold', 'Mask_Blur'])

     # Read information in CSV to parse filepaths for images to analyze
     data.head()

     # Determine number of cases to analyze
     num_cases = data.shape[0]

     time_for_dirname = datetime.datetime.now() 
     root_dirname = str('BIICHT_Outputs_' + str(time_for_dirname)).replace(' ', '_')
     root_dirname = root_dirname.replace(':', '')
     root_dirname = root_dirname.replace('.', '')
     os.mkdir(root_dirname)

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
          mask_px, image_px, image_px_normalized = Masking_to_Average_kernel(image_loc, mask_loc, image_threshold, mask_threshold, mask_blur, root_dirname)
          analysis_outputs.append([image_loc, mask_loc, image_threshold, mask_threshold, mask_blur, int(mask_px), int(image_px), float(image_px_normalized)])
          plt.close()
     
     analysis_outputs_df = pd.DataFrame(analysis_outputs, columns=['Data', 'Mask', 'Data_Threshold', 'Mask_Threshold', 'Mask_Blur', 'Binary_Mask_Px_N', 'Thresholded_Masked_Image_Px_N', 'Thresholded_Masked_Image_Px_N_Normalized'])

     csv_to_save = str(root_dirname + '\\' 'Contrast_statistics_normalized.csv')
     analysis_outputs_df.to_csv(csv_to_save, sep=';', index=False) # Note: For European settings of Excel

def Masking_to_Average_kernel(image_loc_var, mask_loc_var, data_threshold_var, mask_threshold_var, mask_blur_var, root_dirname):
     # Adapted from https://datacarpentry.org/image-processing/07-thresholding.html

     # Load image 
     image_pre_threshold = cv2.imread(image_loc_var)
     #plt.imshow(image_pre_threshold)
     #plt.show()

     # Define threshold for image data and mask by threshold 
     image_file_title_old = image_loc_var.split('\\')[-1]
     image_file_title = image_file_title_old.replace('.', '_')
     image_file_title = image_file_title.replace('-', '_')
     image_file_title = image_file_title.replace(' ', '_')

     #image_threshold = 1
     image_threshold = data_threshold_var
     image_thresholded = Threshold_and_Mask_Image(image_pre_threshold, image_file_title, image_threshold, 'H') 

     # Load mask 
     mask_pre_threshold_unblurred = cv2.imread(mask_loc_var)
     mask_pre_threshold = gaussian_filter(mask_pre_threshold_unblurred, sigma=mask_blur_var)
     #plt.imshow(mask_pre_threshold)
     #plt.show()

     # Define threshold for mask data and mask by threshold 
     mask_file_title_old = mask_loc_var.split('\\')[-1]
     mask_file_title = mask_file_title_old.replace('.', '_')
     mask_file_title = mask_file_title.replace('-', '_')
     mask_file_title = mask_file_title.replace(' ', '_')
     #mask_threshold = 10
     mask_threshold = mask_threshold_var
     mask_thresholded = Threshold_and_Mask_Image(mask_pre_threshold, mask_file_title, mask_threshold, 'H') 
     
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
     plt.savefig(fig_name) 
     plt.close()

     # Apply thresholded mask to thresholded image 
     image_thresholded_and_masked = image_thresholded.copy()
     
     # Convert the thresholded mask to grayscale and then binary 
     mask_thresholded_grayscale = cv2.cvtColor(mask_thresholded, cv2.COLOR_BGR2GRAY)
     ret, mask_thresholded_grayscale_binary_mask = cv2.threshold(mask_thresholded_grayscale, mask_threshold, 255, cv2.THRESH_BINARY)
     
     image_thresholded_and_masked = cv2.bitwise_and(image_thresholded_and_masked, image_thresholded_and_masked, mask = mask_thresholded_grayscale_binary_mask)
     image_thresholded_and_masked_grayscale = cv2.cvtColor(image_thresholded_and_masked, cv2.COLOR_BGR2GRAY)
     #plt.imshow(image_thresholded_and_masked)
     #plt.show() 
     
     fig = plt.figure(figsize=(18, 9)) 

     fig.add_subplot(1, 3, 1) 
     plt.imshow(mask_thresholded_grayscale_binary_mask)
     plt.title('Thresholded Mask as Binary')
     fig.add_subplot(1, 3, 2) 
     plt.imshow(image_thresholded_and_masked)
     plt.title('Thresholded and Masked Image')
     fig.add_subplot(1, 3, 3) 
     plt.imshow(image_thresholded_and_masked)
     plt.imshow(mask_thresholded_grayscale_binary_mask, cmap='jet', alpha=0.5)
     plt.title('Image Result with Mask Overlay')

     fig_name = str(dir_name + '\\Figure_2_Thresholded_Binary_Mask_and_Final_Result.png')
     plt.savefig(fig_name)
     plt.close()  

     # Average the image pixel intensities normalized by mask area 
     mask_number_of_px = mask_thresholded_grayscale_binary_mask[mask_thresholded_grayscale_binary_mask != 0].size # Calculates number of pixels to be considered 
     # image_sum_of_intensities = np.sum(image_thresholded_and_masked) # Calculates sum of pixel intensities
     image_thresholded_and_masked_number_of_px = image_thresholded_and_masked_grayscale[image_thresholded_and_masked_grayscale != 0].size # Note that this has to be converted to greyscale first 
     image_thresholded_and_masked_number_of_px_normalized = image_thresholded_and_masked_number_of_px / mask_number_of_px; 

     return [mask_number_of_px, image_thresholded_and_masked_number_of_px, image_thresholded_and_masked_number_of_px_normalized]

def Threshold_and_Mask_Image(image_to_threshold_var, image_name_var, threshold_value_var, polarity_var): 
     
      # Convert image to grayscale as threshold preprocessing 
     image_pre_threshold_grayscale = cv2.cvtColor(image_to_threshold_var, cv2.COLOR_BGR2GRAY)
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
     image_thresholded_var = cv2.bitwise_and(image_thresholded_var, image_thresholded_var, mask = image_threshold_binary_mask)
     #plt.imshow(image_thresholded_var)
     #plt.show() 

     return image_thresholded_var  

BIICHT_Master()






