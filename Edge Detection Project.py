import numpy as np
import cv2
import matplotlib.pyplot as plt


class EdgeDetectionProject:
    def __init__(self, image_path):
        self.image_path = "images.jpg"
        self.original_image = None
        self.gray_image = None
        self.float_image = None

    def load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Image not found at path {self.image_path}")
        return self.original_image

    def convert_to_grayscale(self):
        if self.original_image is None:
            self.load_image()

        b, g, r = cv2.split(self.original_image)
        self.gray_image = 0.299 * r + 0.587 * g + 0.114 * b
        self.gray_image = self.gray_image.astype(np.uint8)
        return self.gray_image

    def convert_to_float(self):
        if self.gray_image is None:
            self.convert_to_grayscale()

        self.float_image = self.gray_image.astype(np.float32) / 255.0
        return self.float_image

    def manual_convolution2d(self, image, kernel):
        img_h, img_w = image.shape
        kernel_h, kernel_w = kernel.shape

        pad_h = kernel_h // 2
        pad_w = kernel_w // 2

        padded_image = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w), dtype=np.float32)
        padded_image[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = image

        output = np.zeros_like(image, dtype=np.float32)

        for i in range(img_h):
            for j in range(img_w):
                window = padded_image[i:i + kernel_h, j:j + kernel_w]
                output[i, j] = np.sum(window * kernel)

        return output

    def sobel_edge_detection(self, image=None):
        if image is None:
            if self.float_image is None:
                self.convert_to_float()
            image = self.float_image

        Gx_kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=np.float32)

        Gy_kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]], dtype=np.float32)

        Gx = self.manual_convolution2d(image, Gx_kernel)
        Gy = self.manual_convolution2d(image, Gy_kernel)

        magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
        direction = np.arctan2(Gy, Gx)

        return magnitude, direction, Gx, Gy

    def laplacian_edge_detection(self, image=None):
        if image is None:
            if self.float_image is None:
                self.convert_to_float()
            image = self.float_image

        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float32)

        laplacian = self.manual_convolution2d(image, laplacian_kernel)
        return laplacian

    def gaussian_filter(self, image, kernel_size=5, sigma=1.0):
        k = kernel_size // 2
        x, y = np.mgrid[-k:k + 1, -k:k + 1]

        gaussian_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        gaussian_kernel = gaussian_kernel / (2 * np.pi * sigma ** 2)
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

        smoothed = self.manual_convolution2d(image, gaussian_kernel)
        return smoothed

    def non_maximum_suppression(self, magnitude, direction):
        M, N = magnitude.shape
        suppressed = np.zeros_like(magnitude, dtype=np.float32)

        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]

                elif (22.5 <= angle[i, j] < 67.5):
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]

                elif (67.5 <= angle[i, j] < 112.5):
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]

                elif (112.5 <= angle[i, j] < 157.5):
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0

        return suppressed

    def double_threshold(self, image, low_ratio=0.05, high_ratio=0.15):
        high_threshold = image.max() * high_ratio
        low_threshold = high_threshold * low_ratio

        result = np.zeros_like(image, dtype=np.uint8)

        strong = 255
        weak = 50
        zero = 0

        strong_i, strong_j = np.where(image >= high_threshold)
        weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))
        zero_i, zero_j = np.where(image < low_threshold)

        result[strong_i, strong_j] = strong
        result[weak_i, weak_j] = weak
        result[zero_i, zero_j] = zero

        return result, high_threshold, low_threshold

    def hysteresis_edge_tracking(self, image, weak=50, strong=255):
        M, N = image.shape
        result = np.copy(image)

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if result[i, j] == weak:
                    if ((result[i + 1, j - 1] == strong) or (result[i + 1, j] == strong) or
                            (result[i + 1, j + 1] == strong) or (result[i, j - 1] == strong) or
                            (result[i, j + 1] == strong) or (result[i - 1, j - 1] == strong) or
                            (result[i - 1, j] == strong) or (result[i - 1, j + 1] == strong)):
                        result[i, j] = strong
                    else:
                        result[i, j] = 0

        return result

    def manual_canny_edge_detection(self, image=None, sigma=1.0, low_ratio=0.05, high_ratio=0.15):
        if image is None:
            if self.float_image is None:
                self.convert_to_float()
            image = self.float_image

        print("Applying Gaussian filter...")
        smoothed = self.gaussian_filter(image, sigma=sigma)

        print("Step 2: Computing image gradient...")
        magnitude, direction, Gx, Gy = self.sobel_edge_detection(smoothed)

        print("Step 3: Non-maximum suppression...")
        suppressed = self.non_maximum_suppression(magnitude, direction)

        print("Step 4: Double thresholding...")
        thresholded, high_thresh, low_thresh = self.double_threshold(suppressed, low_ratio, high_ratio)

        print("Step 5: Hysteresis edge tracking...")
        final_edges = self.hysteresis_edge_tracking(thresholded)

        return {
            'smoothed': smoothed,
            'gradient_magnitude': magnitude,
            'gradient_direction': direction,
            'non_max_suppressed': suppressed,
            'double_thresholded': thresholded,
            'final_edges': final_edges,
            'Gx': Gx,
            'Gy': Gy,
            'high_threshold': high_thresh,
            'low_threshold': low_thresh
        }

    def visualize_results(self):
        self.load_image()
        self.convert_to_grayscale()
        self.convert_to_float()

        sobel_magnitude, _, _, _ = self.sobel_edge_detection()
        laplacian = self.laplacian_edge_detection()
        canny_results = self.manual_canny_edge_detection()
        canny_manual = canny_results['final_edges']
        canny_opencv = cv2.Canny(self.gray_image, 50, 150)

        sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        laplacian_normalized = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX)

        plt.figure(figsize=(16, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(self.gray_image, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(sobel_normalized, cmap='gray')
        plt.title('Sobel')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(laplacian_normalized, cmap='gray')
        plt.title('Laplacian')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(canny_manual, cmap='gray')
        plt.title('Manual Canny')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(canny_opencv, cmap='gray')
        plt.title('Canny (OpenCV)')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(canny_results['gradient_magnitude'], cmap='gray')
        plt.title('Canny Gradient')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        self.visualize_canny_steps(canny_results)

    def visualize_canny_steps(self, canny_results):
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(canny_results['smoothed'], cmap='gray')
        plt.title('1. Gaussian Filter')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(np.abs(canny_results['Gx']), cmap='gray')
        plt.title('2. Gradient X (Gx)')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(np.abs(canny_results['Gy']), cmap='gray')
        plt.title('3. Gradient Y (Gy)')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(canny_results['gradient_magnitude'], cmap='gray')
        plt.title('4. Gradient Magnitude')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(canny_results['non_max_suppressed'], cmap='gray')
        plt.title('5. Non-max Suppression')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(canny_results['double_thresholded'], cmap='gray')
        plt.title('6. Double Thresholded')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.imshow(canny_results['final_edges'], cmap='gray')
        plt.title('7. Final Edges')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def compare_edge_detectors(self):
        sobel_magnitude, _, _, _ = self.sobel_edge_detection()
        laplacian = self.laplacian_edge_detection()
        canny_results = self.manual_canny_edge_detection()

        sobel_edges = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        laplacian_edges = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX)
        canny_edges = canny_results['final_edges']

        sobel_binary = (sobel_edges > 50).astype(np.uint8) * 255
        laplacian_binary = (laplacian_edges > 30).astype(np.uint8) * 255

        sobel_edge_count = np.sum(sobel_binary > 0)
        laplacian_edge_count = np.sum(laplacian_binary > 0)
        canny_edge_count = np.sum(canny_edges > 0)

        total_pixels = self.gray_image.size
        sobel_edge_ratio = sobel_edge_count / total_pixels * 100
        laplacian_edge_ratio = laplacian_edge_count / total_pixels * 100
        canny_edge_ratio = canny_edge_count / total_pixels * 100

        print("=" * 60)
        print("ANALYSIS AND COMPARISON OF EDGE DETECTION METHODS")
        print("=" * 60)
        print(f"1. Sobel Filter:")
        print(f"   - Number of edge pixels: {sobel_edge_count:,}")
        print(f"   - Edge pixel ratio: {sobel_edge_ratio:.2f}%")
        print(f"   - Features: Simple, Fast, Noise-sensitive")
        print(f"   - Application: General edge detection")
        print()

        print(f"2. Laplacian Filter:")
        print(f"   - Number of edge pixels: {laplacian_edge_count:,}")
        print(f"   - Edge pixel ratio: {laplacian_edge_ratio:.2f}%")
        print(f"   - Features: Noise-sensitive, Thin edges")
        print(f"   - Application: Precise edge detection in low-noise images")
        print()

        print(f"3. Canny Algorithm:")
        print(f"   - Number of edge pixels: {canny_edge_count:,}")
        print(f"   - Edge pixel ratio: {canny_edge_ratio:.2f}%")
        print(f"   - High threshold: {canny_results['high_threshold']:.4f}")
        print(f"   - Low threshold: {canny_results['low_threshold']:.4f}")
        print(f"   - Features: Accurate, Noise-resistant, Continuous edges")
        print(f"   - Application: Precision applications like object detection")
        print("=" * 60)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(sobel_magnitude.flatten(), bins=50, alpha=0.7, color='blue')
        plt.title('Sobel Gradient Distribution')
        plt.xlabel('Gradient Magnitude')
        plt.ylabel('Pixel Count')

        plt.subplot(1, 3, 2)
        plt.hist(np.abs(laplacian.flatten()), bins=50, alpha=0.7, color='green')
        plt.title('Laplacian Magnitude Distribution')
        plt.xlabel('Magnitude Value')
        plt.ylabel('Pixel Count')

        plt.subplot(1, 3, 3)
        plt.bar(['Sobel', 'Laplacian', 'Canny'],
                [sobel_edge_ratio, laplacian_edge_ratio, canny_edge_ratio],
                color=['blue', 'green', 'red'], alpha=0.7)
        plt.title('Edge Pixel Ratio Comparison')
        plt.ylabel('Edge Pixel Percentage (%)')

        plt.tight_layout()
        plt.show()


def main():
    print("EDGE DETECTION PROJECT - IMAGE PROCESSING")
    print("=" * 50)

    image_path = "input_image.jpg"

    try:
        edge_detector = EdgeDetectionProject(image_path)
        edge_detector.load_image()
    except:
        print("Sample image not found. Creating synthetic image...")
        synthetic_image = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(synthetic_image, (50, 50), (200, 200), 255, -1)
        cv2.circle(synthetic_image, (300, 100), 50, 255, -1)
        cv2.line(synthetic_image, (100, 300), (300, 350), 255, 3)

        cv2.imwrite("synthetic_image.jpg", synthetic_image)
        image_path = "synthetic_image.jpg"
        edge_detector = EdgeDetectionProject(image_path)

    print("1. Loading and preprocessing image...")
    edge_detector.load_image()
    gray = edge_detector.convert_to_grayscale()
    float_img = edge_detector.convert_to_float()
    print(f"   - Image dimensions: {gray.shape}")
    print(f"   - Grayscale image data type: {gray.dtype}")
    print(f"   - Float image data type: {float_img.dtype}")
    print()

    print("2. Testing 2D convolution function...")
    test_kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    test_result = edge_detector.manual_convolution2d(float_img, test_kernel)
    print(f"   - Convolution output dimensions: {test_result.shape}")
    print(f"   - Output mean value: {np.mean(test_result):.4f}")
    print()

    print("3. Edge detection with different methods...")

    print("   - Applying Sobel filter...")
    sobel_mag, sobel_dir, Gx, Gy = edge_detector.sobel_edge_detection()
    print(f"     Gradient magnitude: min={sobel_mag.min():.4f}, max={sobel_mag.max():.4f}")

    print("   - Applying Laplacian filter...")
    laplacian = edge_detector.laplacian_edge_detection()
    print(f"     Laplacian output: min={laplacian.min():.4f}, max={laplacian.max():.4f}")

    print("   - Applying Canny algorithm (manual implementation)...")
    canny_results = edge_detector.manual_canny_edge_detection()
    print(f"     Thresholds: high={canny_results['high_threshold']:.4f}, low={canny_results['low_threshold']:.4f}")
    print()

    print("4. Displaying results...")
    edge_detector.visualize_results()

    print("5. Analysis and comparison of methods...")
    edge_detector.compare_edge_detectors()

    print("\nProject completed successfully!")
    print("Results displayed in separate windows.")


if __name__ == "__main__":
    main()