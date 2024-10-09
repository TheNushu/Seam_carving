%% Seam Carving for Content-Aware Image Resizing
% This MATLAB script implements the seam carving algorithm for intelligent
% image resizing (cropping or expanding) as described in:
% https://en.wikipedia.org/wiki/Seam_carving and in the following code
% documentation area.
%
% Author: Daniel-Ioan Mlesnita
% Date: 02.10.2024
% Version: 1.0
%
% Usage:
%   1. Ensure the image file is in the same directory as this script.
%   2. Run the script in MATLAB.
%   3. When prompted, enter the scale factor (between 0 and 1.99).
%      - Values between 0 and 1 will reduce the image width.
%      - Values between 1 and 1.99 will expand the image width.
%      - A value of 1 will not change the image size.
%
% Input:
%   - Image file: e.g. 'prague.jpg'
%   - Scale factor: Entered by user during runtime
%
% Output:
%   - Displays a figure with four subplots:
%     1. Original Image
%     2. Grayscale Energy Map
%     3. Original Image with Processed Seams
%     4. Processed (Resized) Image
%   - Saves the processed image as 'processed_image.jpg' in the same directory
%
% Real-time Progress:
%   The script provides real-time updates on the progress of the seam carving
%   process in the MATLAB Command Window. It displays the percentage of seams
%   processed for both removal and insertion operations.
%
% Performance Characteristics:
%   The computational complexity of the seam carving process is approximately
%   O(w * h * |1 - scale_factor|), where w and h are the width and height of 
%   the image, respectively. This means the processing time increases linearly 
%   with the image resolution and the amount of resizing required.
%   
%   However, for very large images or extreme scaling factors, the process
%   might become noticeably slower due to increased memory requirements and
%   the cumulative effect of multiple seam operations.
%
% Requirements:
%   - MATLAB Image Processing Toolbox
%
% Note: The quality of results may vary depending on the image content,
% image size and the chosen scale factor. At the current version, artifacts
% might become visible even at small scale factors. Additional features,
% such as selecting areas to have higher "energy", might provide better
% results, by being less likely to be chosen and seam paths.

% Main script
image = imread('prague.jpg');

% Scale factor decides how much of the original image do you want to keep
% or expand to
% e.g. 0.35 = reduce to 35% of original picture 
% e.g. 1.35 = expand to 135% of original picture

scale_factor = getValidScaleFactor();

[processedImage, seamImage, energyMap] = seamCarving(image, scale_factor);

displayResults(image, processedImage, seamImage, energyMap);

% Save processed image
imwrite(processedImage, 'processed_image.jpg');

% Function to get a valid scale factor from the user
function scale_factor = getValidScaleFactor()
    while true
        scale_factor = input('Enter scale factor (between 0 and 1.99): ');
        if scale_factor >= 0 && scale_factor < 2
            break;
        else
            fprintf('Error: Scale factor must be between 0 and 1.99. Please try again.\n');
        end
    end
end

% display original, energy map, original_with_seams, and processed images
function displayResults(originalImage, processedImage, seamImage, energyMap)
    figure;
    
    % Top row: Original Image, Energy Map, and Original Image with Processed Seams
    subplot(2, 3, 1);
    imshow(originalImage);
    title('Original Image');
    
    subplot(2, 3, 2);
    imshow(energyMap);
    title('Grayscale Energy Map');
    
    subplot(2, 3, 3);
    imshow(seamImage);
    title('Original Image with Processed Seams');
    
    % Bottom row: Processed Image (spanning all three columns)
    subplot(2, 3, [4, 5, 6]);
    imshow(processedImage);
    title('Processed Image');
    
    % Adjust the overall figure size for better visibility
    set(gcf, 'Position', get(0, 'Screensize'));
end

% Main seam carving function

% Inputs: image (input image), scale_factor (desired scaling)
% Outputs: output (processed image),
% seamImage (original image with highlighted seams),
% energyMap (energy map of the original image)
% Process:  Handles both image reduction and expansion based on the scale
% factor by calls to the functions: reduceImage and expandImage
function [output, seamImage, energyMap] = seamCarving(image, scale_factor)
    % Get width of image to get the proportional number of seams
    % Change to [h, ~, ~] if you want to change the image based on height
    % don't forget to edit w to h later on in the function for that
    [~, w, ~] = size(image);
    
    % Compute initial energy map
    if size(image, 3) == 3
        grayImage = rgb2gray(image);
    else
        grayImage = image;
    end
    energyMap = energyFunction(grayImage);
    
    if scale_factor < 1
        % Reduction mode
        % Get number of seams proportional to image width
        numSeams = round(w * (1 - scale_factor));
        [output, seamImage] = reduceImage(image, numSeams);
    elseif scale_factor > 1
        % Expansion mode
        % Get number of seams proportional to image width
        numSeams = round(w * (scale_factor - 1));
        [output, seamImage] = expandImage(image, numSeams);
    else
        % No change
        output = image;
        seamImage = image;
    end
    fprintf('Seam carving complete. Final image width: %d pixels (%.2f%% of original)\n', ...
        size(output, 2), (size(output, 2) / w) * 100);
end

% Function to reduce image size by removing seams
% Inputs: image (input image), numSeams (number of seams to remove)
% Outputs: output (reduced image), seamImage (original image with highlighted seams)
% Process: Iteratively removes vertical seams from the image by calculating
% the energy map, finding the optimal seam, highlighting it, and then
% removing it from the image.
function [output, seamImage] = reduceImage(image, numSeams)
    % Initialize seamImage as a copy of the original image
    seamImage = image;
    
    % Initialize a mask to keep track of shifted pixels
    [h, w, ~] = size(image);
    mask = ones(h, w);
    
    % Store all seams to be removed
    allSeams = cell(numSeams, 1);
    
    % Find and store all seams first
    tempImage = image;
    tempMask = mask;
    for i = 1:numSeams
        if size(tempImage, 3) == 3
            grayImage = rgb2gray(tempImage);
        else
            grayImage = tempImage;
        end
        
        energyMap = energyFunction(grayImage);
        seam = findVerticalSeam(energyMap);
        allSeams{i} = seam;
        
        % Highlight seam in seamImage
        seamImage = highlightSeam(seamImage, seam, tempMask);
        
        % Remove seam temporarily from tempImage
        [tempImage, tempMask] = removeSeam(tempImage, seam, tempMask);
        
        fprintf('Found seam %d of %d (%.2f%%)\n', i, numSeams, (i/numSeams)*100);
    end
    
    % Remove all stored seams
    for i = 1:numSeams
        seam = allSeams{i};
        [image, mask] = removeSeam(image, seam, mask);
        fprintf('Removed seam %d of %d (%.2f%%)\n', i, numSeams, (i/numSeams)*100);
    end
    
    output = image;
end

% Function to expand image size by inserting seams
% Inputs: image (input image), numSeams (number of seams to insert)
% Outputs: output (expanded image), seamImage (original image with highlighted seams)
% Process: First finds and stores multiple seams in the original image, then 
% inserts these seams one by one to expand the image.
function [output, seamImage] = expandImage(image, numSeams)
    % Initialize seamImage as a copy of the original image
    seamImage = image;
    % Initialize a mask to keep track of shifted pixels
    [h, w, ~] = size(image);
    mask = ones(h, w);
    % Store all seams to be inserted
    allSeams = cell(numSeams, 1);
    
    % Find and store all seams first
    tempImage = image;
    tempMask = mask;
    for i = 1:numSeams
        if size(tempImage, 3) == 3
            grayImage = rgb2gray(tempImage);
        else
            grayImage = tempImage;
        end
        
        energyMap = energyFunction(grayImage);
        seam = findVerticalSeam(energyMap);
        allSeams{i} = seam;
        
        % Highlight seam in seamImage
        seamImage = highlightSeam(seamImage, seam, tempMask);
        
        % Remove seam temporarily from tempImage
        [tempImage, tempMask] = removeSeam(tempImage, seam, tempMask);
        
        fprintf('Found seam %d of %d (%.2f%%)\n', i, numSeams, (i/numSeams)*100);
    end
    
    % Insert all stored seams
    for i = 1:numSeams
        seam = allSeams{i};
        [image, mask] = insertSeam(image, seam, mask);
        fprintf('Inserted seam %d of %d (%.2f%%)\n', i, numSeams, (i/numSeams)*100);
    end
    
    output = image;
end

% Function to insert a seam into the image
% Inputs: image (input image), seam (seam to insert), mask (current mask)
% Outputs: output (image with inserted seam), newMask (updated mask)
% Process: Inserts a new seam into the image by shifting pixels and interpolating 
% new pixel values along the seam path.
function [output, newMask] = insertSeam(image, seam, mask)
    [h, w, d] = size(image);
    output = zeros(h, w+1, d, 'like', image);
    newMask = zeros(h, w+1);
    for i = 1:h
        col = seam(i);
        output(i, 1:col, :) = image(i, 1:col, :);
        if col < w
            output(i, col+1, :) = round((double(image(i, col, :)) + double(image(i, col+1, :))) / 2);
        else
            output(i, col+1, :) = image(i, col, :);
        end
        output(i, col+2:end, :) = image(i, col+1:end, :);
        
        newMask(i, 1:col) = mask(i, 1:col);
        newMask(i, col+1) = 1;
        newMask(i, col+2:end) = mask(i, col+1:end);
    end
end

% Function to compute the energy map of an image
% Input: image (grayscale image)
% Output: energy (energy map of the image)
% Process: Computes the energy map of an image by applying Gaussian smoothing, 
% calculating gradients using the Sobel operator, and normalizing the result.
function energy = energyFunction(image)
    % parameter responsible for edge smoothening/noise reduction
    % You'd want:
    % high sigma for smooth "edges" and less noise
    % low sigma to perserve more information

    % Recommended range: 0.25 - 1.5
    sigma = 0.75;
    % Apply Gaussian smoothening
    smoothed_image = imgaussfilt(image, sigma);
    
    % Compute the gradient using Sobel operator
    [Gx, Gy] = imgradientxy(smoothed_image, 'sobel');
    
    % Compute the gradient magnitude
    gradient_magnitude = sqrt(double(Gx).^2 + double(Gy).^2);
    
    % Normalize the gradient magnitude to [0, 1] range
    energy = mat2gray(gradient_magnitude);
    
    % Apply contrast stretching (optional)
    % energy = imadjust(energy);
end

% Function to find the vertical seam with minimum energy
% Input: energyMap (energy map of the image)
% Output: seam (vertical seam with minimum energy)
% Process: Uses dynamic programming to find the vertical path through the 
% image with the lowest total energy.
function seam = findVerticalSeam(energyMap)
    [h, w] = size(energyMap);
    % Initialize cumulative energy map
    M = energyMap;
    % Compute cumulative minimum energy
    for i = 2:h
        for j = 1:w
            if j == 1
                M(i,j) = M(i,j) + min(M(i-1,j), M(i-1,j+1));
            elseif j == w
                M(i,j) = M(i,j) + min(M(i-1,j-1), M(i-1,j));
            else
                M(i,j) = M(i,j) + min([M(i-1,j-1), M(i-1,j), M(i-1,j+1)]);
            end
        end
    end
    % Backtrack to find the seam
    seam = zeros(h, 1);
    [~, seam(h)] = min(M(h, :));
    for i = h-1:-1:1
        if seam(i+1) == 1
            [~, idx] = min(M(i, 1:2));
        elseif seam(i+1) == w
            [~, idx] = min(M(i, w-1:w));
            idx = idx + w - 2;
        else
            [~, idx] = min(M(i, seam(i+1)-1:seam(i+1)+1));
            idx = idx + seam(i+1) - 2;
        end
        seam(i) = idx;
    end
end

% Function to remove a seam from the image
% Inputs: image (input image), seam (seam to remove), mask (current mask)
% Outputs: output (image with removed seam), newMask (updated mask)
% Process: Removes a specified seam (2nd parameter) from the image by
% shifting pixels to fill the gap left by the removed seam.
function [output, newMask] = removeSeam(image, seam, mask)
    % Get the dimensions of the input image
    [h, w, d] = size(image);
    % Initialize output image with one less column than the input
    output = zeros(h, w-1, d, 'like', image);
    % Initialize new mask with one less column
    newMask = zeros(h, w-1);

    % Iterate through each row of the image
    for i = 1:h
        % Copy pixels before the seam
        output(i, 1:seam(i)-1, :) = image(i, 1:seam(i)-1, :);        
        % Copy pixels after the seam, shifting them left
        output(i, seam(i):end, :) = image(i, seam(i)+1:end, :);        
        % Update mask before the seam
        newMask(i, 1:seam(i)-1) = mask(i, 1:seam(i)-1);        
        % Update mask after the seam, shifting it left
        newMask(i, seam(i):end) = mask(i, seam(i)+1:end);
    end
end

% Function to highlight a seam in the image
% Inputs: seamImage (image to highlight seam in), seam (seam to highlight),
% mask (current mask)
% Output: seamImage (image with highlighted seam)
% Process: Marks a specified seam in the image by coloring its pixels with 
% a bright color (yellow for color images, white for grayscale).
function seamImage = highlightSeam(seamImage, seam, mask)
    [h, ~, d] = size(seamImage);
    % Choose a bright color for highlighting (e.g., yellow)
    highlightColor = [255, 255, 0];
    for i = 1:h
        % Find the actual position in the original image
        originalPos = find(cumsum(mask(i, :)) == seam(i), 1);
        if d == 1 % Grayscale image
            seamImage(i, originalPos) = 255; % White for grayscale
        else % Color image
            seamImage(i, originalPos, :) = highlightColor;
        end
    end
end