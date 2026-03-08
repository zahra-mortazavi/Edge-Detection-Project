# Edge Detection Project

## Overview

This project demonstrates the implementation of classic edge detection algorithms in image processing using Python. It includes manual implementations of the Sobel filter, Laplacian filter, and Canny algorithm without relying on high-level library functions for core operations. The project processes an input image (grayscale conversion, floating-point normalization), applies edge detection methods, visualizes intermediate steps (especially for Canny), and provides a quantitative comparison of the results.

The code is structured as a class `EdgeDetectionProject` with methods for each step, and a `main()` function to run the pipeline. If the input image ("input_image.jpg") is not found, a synthetic image is generated for testing.

![images](https://github.com/user-attachments/assets/04882da0-243e-4733-ab77-177b233e1c03)

### Key Algorithms Implemented
- **Sobel Filter**: Detects edges by computing image gradients in X and Y directions using convolution kernels.
- **Laplacian Filter**: Uses a second-order derivative kernel to highlight regions of rapid intensity change.
- **Canny Algorithm (Manual)**: A multi-stage process including Gaussian smoothing, gradient computation (Sobel-based), non-maximum suppression, double thresholding, and hysteresis edge tracking.
- Additional utilities: Manual 2D convolution, image loading/preprocessing, visualization, and performance analysis.

## Requirements
- Python 3.6+
- NumPy (`pip install numpy`)
- OpenCV (`pip install opencv-python`)
- Matplotlib (`pip install matplotlib`)

## Installation
1. Clone the repository:
   ```
   git clone <repo-url>
   cd edge-detection-project
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Create `requirements.txt` with the libraries above if needed.)

3. Place your input image as `input_image.jpg` in the project directory (or use the synthetic fallback).

## Usage
Run the script:
```
python edge_detection.py
```
- The script will:
  1. Load and preprocess the image.
  2. Apply and test convolution.
  3. Perform edge detection with Sobel, Laplacian, and manual Canny.
  4. Display visualizations (results grid, Canny steps, final edges, histograms, and bar charts).
  5. Print analysis and comparison in the console.

Example console output during execution:
```
EDGE DETECTION PROJECT - IMAGE PROCESSING
==================================================
1. Loading and preprocessing image...
 - Image dimensions: (height, width)
 - Grayscale image data type: uint8
 - Float image data type: float32

2. Testing 2D convolution function...
 - Convolution output dimensions: (height, width)
 - Output mean value: X.XXXX

3. Edge detection with different methods...
 - Applying Sobel filter...
 Gradient magnitude: min=X.XXXX, max=X.XXXX
 - Applying Laplacian filter...
 Laplacian output: min=X.XXXX, max=X.XXXX
 - Applying Canny algorithm (manual implementation)...
 Applying Gaussian filter...
 Step 2: Computing image gradient...
 Step 3: Non-maximum suppression...
 Step 4: Double thresholding...
 Step 5: Hysteresis edge tracking...
 Thresholds: high=X.XXXX, low=X.XXXX

4. Displaying results...
(Plots will appear in separate windows)

5. Analysis and comparison of methods...
(Printed analysis below)

Project completed successfully!
Results displayed in separate windows.
```

## Code Structure
- **`__init__`**: Initializes with image path.
- **`load_image`**: Loads the RGB image using OpenCV.
- **`convert_to_grayscale`**: Converts to grayscale using weighted RGB channels.
- **`convert_to_float`**: Normalizes grayscale to [0,1] float32.
- **`manual_convolution2d`**: Custom 2D convolution with padding.
- **`sobel_edge_detection`**: Computes Gx, Gy, magnitude, and direction.
- **`laplacian_edge_detection`**: Applies Laplacian kernel.
- **`gaussian_filter`**: Generates and applies Gaussian kernel.
- **`non_maximum_suppression`**: Thins edges based on gradient direction.
- **`double_threshold`**: Applies high/low thresholds, marking strong/weak edges.
- **`hysteresis_edge_tracking`**: Connects weak edges to strong ones.
- **`manual_canny_edge_detection`**: Full Canny pipeline with configurable sigma and ratios.
- **`visualize_results`**: Plots input, Sobel, Laplacian, Manual Canny, OpenCV Canny, and Gradient.
- **`visualize_canny_steps`**: Plots each Canny step.
- **`compare_edge_detectors`**: Computes edge counts/ratios, prints analysis, and plots distributions/comparisons.
- **`main`**: Orchestrates the process, handles synthetic image fallback.

## Results and Visualizations
The project was tested on a grayscale image of a daisy flower (dimensions approximately 400x300). Below are descriptions of the key outputs. (In a real repository, save plots as PNG files using `plt.savefig()` and link them here.)

### 1. Main Results Grid
A 2x3 grid showing:
- **Input Image**: Original grayscale daisy flower.
- **Sobel**: Thick, noisy edge outline of the flower petals and stem.
- **Laplacian**: Thinner, more precise but noise-sensitive edges.
- **Manual Canny**: Spiky, detailed edges with some discontinuities.
- **Canny (OpenCV)**: Smoother, continuous edges (for comparison).
- **Canny Gradient**: Gradient magnitude similar to Sobel.

<img width="1230" height="771" alt="Screenshot 2026-03-08 042522" src="https://github.com/user-attachments/assets/94d4563c-5838-4b60-be5a-f084dbf1b7ca" />

### 2. Canny Algorithm Steps Grid
A 2x3 grid illustrating the manual Canny process:
- **1. Gaussian Filter**: Blurred version of the input to reduce noise.
- **2. Gradient X (Gx)**: Horizontal edges (e.g., stem sides).
- **3. Gradient Y (Gy)**: Vertical edges (e.g., petal tops/bottoms).
- **4. Gradient Magnitude**: Combined intensity changes.
- **5. Non-max Suppression**: Thinned edges.
- **6. Double Thresholded**: Edges marked as strong (white), weak (gray), or none (black).

<img width="1233" height="767" alt="Screenshot 2026-03-08 042543" src="https://github.com/user-attachments/assets/084e3b49-caaf-48d2-be02-b969b2a2668e" />

### 3. Final Edges (Hysteresis Output)
The refined Canny edges after connecting weak to strong pixels: A clean white outline of the daisy on a black background.

<img width="999" height="739" alt="Screenshot 2026-03-08 042616" src="https://github.com/user-attachments/assets/af9ed978-6d7f-48ef-b2de-804ca9f212af" />

### 4. Analysis and Comparison
Quantitative metrics computed on binary edge maps (thresholds applied for Sobel/Laplacian):
```
===========================================================
ANALYSIS AND COMPARISON OF EDGE DETECTION METHODS
===========================================================
1. Sobel Filter:
 - Number of edge pixels: 2,923
 - Edge pixel ratio: 5.81%
 - Features: Simple, Fast, Noise-sensitive
 - Application: General edge detection

2. Laplacian Filter:
 - Number of edge pixels: 2,665
 - Edge pixel ratio: 5.30%
 - Features: Noise-sensitive, Thin edges
 - Application: Precise edge detection in low-noise images

3. Canny Algorithm:
 - Number of edge pixels: 894
 - Edge pixel ratio: 1.78%
 - High threshold: 0.3464
 - Low threshold: 0.0173
 - Features: Accurate, Noise-resistant, Continuous edges
 - Application: Precision applications like object detection
===========================================================
```

### 5. Distributions and Comparison Plots
- **Sobel Gradient Distribution**: Histogram showing most pixels near 0, with a tail up to ~4.
- **Laplacian Magnitude Distribution**: Similar, concentrated near 0, tail up to ~2.5.
- **Edge Pixel Ratio Comparison**: Bar chart with Sobel (blue, ~5.8%), Laplacian (green, ~5.3%), Canny (red, ~1.8%).

<img width="1342" height="437" alt="Screenshot 2026-03-08 042640" src="https://github.com/user-attachments/assets/7b2901ae-ae3f-4011-9320-d7512a0f8b74" />

## Analysis Insights
- **Sobel**: Detects more edges but is sensitive to noise, leading to thicker boundaries.
- **Laplacian**: Produces thinner edges but amplifies noise, suitable for clean images.
- **Canny**: Most accurate and noise-resistant, with fewer false positives, ideal for real-world applications like object detection. The manual implementation closely matches OpenCV's but may differ slightly due to custom thresholds.

The edge ratios indicate Canny is more selective (lower % of edge pixels), focusing on strong, continuous features.

## Limitations
- Manual implementations are slower than optimized library functions.
- Assumes grayscale input; color images are converted.
- Thresholds are hardcoded but configurable in methods.
- No GPU acceleration.

## Future Improvements
- Add support for color edge detection.
- Implement adaptive thresholding.
- Add CLI arguments for image path, sigma, thresholds.
- Save outputs as images instead of showing.

