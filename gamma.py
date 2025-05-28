import cv2
import numpy as np
import matplotlib.pyplot as plt

#1) Purpose: Performs gamma correction on an image and allows additional adjustments for contrast and brightness.
def enhanced_gamma_correction(image, gamma, contrast=1.0, brightness=0):
    """
    Improved gamma correction with additional contrast and brightness parameters
    
    Args:
        image: input image (8-bit)
        gamma: gamma correction parameter
        contrast: contrast coefficient (default 1.0)
        brightness: brightness value (default 0)
    
    Returns:
        Image after transformations
    """

    
    # # Normalization and gamma correction
    # Create a "Look-Up Table" (LUT)
    # For each possible pixel value (0-255), calculate a new value
    look_up_table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using the LUT
    # This is faster than applying the formula to each pixel individually
    gamma_corrected = cv2.LUT(image, look_up_table)
    
    # alpha is contrast, beta is brightness
    #cv2.convertScaleAbs  ensures pixel values stay within the valid [0, 255] range.
    adjusted = cv2.convertScaleAbs(gamma_corrected, alpha=contrast, beta=brightness)
    
    return adjusted

#2) Purpose: To cut out a small piece (template) of a specified size from a larger image.
def smart_template_extraction(image, template_size=(60, 60), method='center'):
    """
    Intelligent template extraction with a choice of strategy
    
    Args:
        image: source image
        template_size: template dimensions (width, height)
        method: selection strategy ('center', 'random', 'salient')
    
    Returns:
        extracted_template: the extracted template
        position: (x,y) coordinates of the top-left corner of the template
    """
    h, w = image.shape[:2] # Get height (h) and width (w) of the image
    tw, th = template_size # Get width (tw) and height (th) of the desired template
    
    if method == 'center':
        # Crop the template from the center of the image
        x = (w - tw) // 2
        y = (h - th) // 2
    elif method == 'random':
         # Crop the template from a random location
        x = np.random.randint(0, w - tw + 1) # +1 to include the right/bottom boundary
        y = np.random.randint(0, h - th + 1)
    elif method == 'salient':
        #  # cv2.goodFeaturesToTrack finds strong corners in the image (Shi-Tomasi method)
        corners = cv2.goodFeaturesToTrack(image, 10, 0.01, 10)
        if corners is not None:
            # Take the coordinates of the first found corner
            x, y = map(int, corners[0][0])
            x = max(0, min(x, w - tw))
            y = max(0, min(y, h - th))
        else:
            # If no corners are found, crop from the center (fallback)
            x = (w - tw) // 2
            y = (h - th) // 2
    # Crop the template from the image using NumPy slicing
    template = image[y:y+th, x:x+tw]
    return template, (x, y)

#Finds the template within the larger image.
def advanced_template_matching(image, template, threshold=0.7):
    """
    Improved template matching with multiple methods and visualization
    
    Args:
        image: source image to search in
        template: template to search for
        threshold: correlation threshold (how similar it must be to be considered a match)
    
    Returns:
        result_image: image with marked matches
        max_val: maximum correlation value (best match)
        heatmap: correlation heatmap
    """
    # List of comparison methods we will try
    # TM_CCOEFF_NORMED and TM_CCORR_NORMED are normalized methods,
    # robust to changes in brightness and contrast.
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    results = []
    
    for method in methods:
        # cv2.matchTemplate "slides" the template across the entire image
        # and calculates a similarity coefficient for each position.
        # The result (res) is a map (array) of these coefficients.
        res = cv2.matchTemplate(image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        results.append((res, max_val, max_loc))
    
    # Select the best result from all tried methods
    # (the one with the highest maximum similarity coefficient 'max_val')
    best_idx = np.argmax([r[1] for r in results])
    res, max_val, max_loc = results[best_idx]
    
    # Create a heatmap
    heatmap = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = template.shape
    
    # Find all locations where the similarity coefficient is greater than or equal to 'threshold'
    loc = np.where(res >= threshold)
    # Draw green rectangles around these locations
    for pt in zip(*loc[::-1]):  # pt is (x, y) of the top-left corner
        cv2.rectangle(result_image, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)
    
    
    cv2.rectangle(result_image, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,0,255), 2)
    
    return result_image, max_val, heatmap

def main():
    # Load the image
    img_path = 'BONES.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)# Read the image directly in grayscale
    
    if img is None:
        print(f"Error loading image {img_path}")
        return
    
    # Example of resizing
    img = cv2.resize(img, (800, 600))
    
    # 1. Gamma correction with different parameters
    gammas = [0.3, 0.7, 1.0, 1.5, 2.5]
    corrected_images = [enhanced_gamma_correction(img, g, 1.2, 10) for g in gammas]
    
    # 2. Template extraction
    template, pos = smart_template_extraction(img, (70, 70), 'salient')
    
    # 3. Template matching
    # Threshold 0.65
    matched_img, max_corr, heatmap = advanced_template_matching(img, template, 0.65)
    
    # Вvisualization of results
    plt.figure(figsize=(18, 12))
    
    
    plt.subplot(331), plt.imshow(img, cmap='gray'), plt.title('The original image')
    plt.subplot(332), plt.imshow(template, cmap='gray'), plt.title(f'Extracted Template (position {pos})')
    
    # Gamma correction
    plt.subplot(333), plt.imshow(corrected_images[0], cmap='gray'), plt.title(f'Gamma={gammas[0]} (contrast 1.2)')
    plt.subplot(334), plt.imshow(corrected_images[1], cmap='gray'), plt.title(f'Gamma={gammas[1]} (contrast 1.2)')
    plt.subplot(335), plt.imshow(corrected_images[2], cmap='gray'), plt.title(f'Gamma={gammas[2]} (without changes)')
    plt.subplot(336), plt.imshow(corrected_images[3], cmap='gray'), plt.title(f'Gamma={gammas[3]} (contrast 1.2)')
    
    # Correlation
    plt.subplot(337), plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)), plt.title(f'Correlation heat map (max={max_corr:.3f})')
    plt.subplot(338), plt.imshow(matched_img), plt.title('Search Results(green > 0.65, red - the best)')
    
    # Saving the results
    cv2.imwrite('gamma_0.3.jpg', corrected_images[0])
    cv2.imwrite('gamma_1.0.jpg', corrected_images[2])
    cv2.imwrite('gamma_2.5.jpg', corrected_images[4])
    cv2.imwrite('template.jpg', template)
    cv2.imwrite('correlation_result.jpg', matched_img)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()