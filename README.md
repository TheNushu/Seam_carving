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

## Author
Daniel-Ioan Mlesnita

## Date
October 2, 2024

## Version
1.0

## Acknowledgements
Inspired by the paper "Seam Carving for Content-Aware Image Resizing" by Shai Avidan and Ariel Shamir.
