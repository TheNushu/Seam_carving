# Seam Carving for Content-Aware Image Resizing

## Overview
This project implements the seam carving algorithm for intelligent image resizing using MATLAB. It allows for content-aware resizing of images, preserving important features while reducing or expanding image dimensions.

## Features
- Resize images (reduce or expand) while preserving key content
- Visualize energy maps and seam paths
- Support for both color and grayscale images
- Real-time progress updates during processing

## Requirements
- MATLAB
- Image Processing Toolbox

## Usage
1. Ensure the image file is in the same directory as the script.
2. Run `seam_carving.m` in MATLAB.
3. When prompted, enter a scale factor:
   - Between 0 and 1 to reduce image width
   - Between 1 and 1.99 to expand image width
   - 1 to keep the original size

## Output
- Displays four subplots:
  1. Original Image
  2. Grayscale Energy Map
  3. Original Image with Processed Seams
  4. Processed (Resized) Image
- Saves the processed image as 'processed_image.jpg'

## Performance
- Complexity: O(w * h * |1 - scale_factor|)
- Processing time increases linearly with image resolution and resizing amount

## Limitations
- May produce visible artifacts, especially with large scale factors
- Performance may degrade with very large images

## Future Work
- Implement user-defined areas of importance
- Explore alternative energy functions for improved results

## Implementation Date
October, 2024

## Version
1.0

## Acknowledgements
Inspired by the paper "Seam Carving for Content-Aware Image Resizing" by Shai Avidan and Ariel Shamir.

## Example of output (please check the report for a better view)

   1 - Original Image , 2 - Grayscale Energy Map, 3 - Original Image with Processed Seams, 4 - Processed (Resized) 85% Image 

   1. <img src="https://github.com/user-attachments/assets/a4092a72-f466-421b-aebd-81f4b17ba617" width="25%" height="25%" alt="field">
   2. <img src="https://github.com/user-attachments/assets/fd56fafc-20bb-4cf0-98f4-0d17c63ebbf8" width="25%" height="25%" alt="field">
   3. <img src="https://github.com/user-attachments/assets/2516acb6-09d0-4a22-8bf2-679442bdfbf3" width="25%" height="25%" alt="field">
   4. <img src="https://github.com/user-attachments/assets/f319a6aa-0136-49a8-8299-fc212cc9e54e" width="25%" height="25%" alt="field">


