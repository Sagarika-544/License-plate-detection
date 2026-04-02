# =============================================================================
# src/detection/preprocessor.py
# Smart License Plate Detection System
# Prepares raw images/frames before passing them to the YOLO detector.
# =============================================================================

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_SIZE     = (640, 640)   # Standard YOLOv8 input resolution
NORMALIZE_MEAN   = 0.0          # Shift pixel values to [0, 1] range
NORMALIZE_SCALE  = 1 / 255.0


# =============================================================================
# INDIVIDUAL FUNCTIONS  (use standalone or via preprocess_frame())
# =============================================================================

def resize_image(image, target_size: tuple = DEFAULT_SIZE) -> np.ndarray:
    """
    Resize an image to the target (width, height) using INTER_LINEAR interpolation.
    INTER_LINEAR is fast and produces good quality for upscaling/downscaling.

    Args:
        image       : BGR numpy array.
        target_size : (width, height) tuple. Default matches YOLOv8 input size.

    Returns:
        Resized BGR numpy array.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] → [0.0, 1.0].
    Neural networks converge faster and more stably on normalized inputs.

    Args:
        image : BGR uint8 numpy array.

    Returns:
        float32 numpy array with values in [0.0, 1.0].
    """
    return (image.astype(np.float32)) * NORMALIZE_SCALE


def denoise(image) -> np.ndarray:
    """
    Remove noise using a fast bilateral filter.
    Bilateral filtering smooths noise while preserving sharp edges (plate borders).

    Args:
        image : BGR numpy array.

    Returns:
        Denoised BGR numpy array.
    """
    # d=9: neighbourhood diameter | sigmaColor/sigmaSpace=75: smoothing strength
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def sharpen(image) -> np.ndarray:
    """
    Apply an unsharp-mask sharpening kernel to enhance plate character edges.
    Especially helpful when input is blurry (moving vehicles, low-res cameras).

    Args:
        image : BGR numpy array.

    Returns:
        Sharpened BGR numpy array.
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def equalize_brightness(image) -> np.ndarray:
    """
    Normalize brightness and contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalization) on the Luminance channel of the LAB color space.

    This handles:
      - Overexposed plates (bright sunlight)
      - Underexposed plates (night, shadows)

    Args:
        image : BGR numpy array.

    Returns:
        Brightness-equalized BGR numpy array.
    """
    # Convert BGR → LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split into channels; we only equalize L (lightness)
    l_channel, a, b = cv2.split(lab)

    # CLAHE: clipLimit prevents over-amplification; tileGridSize sets region size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)

    # Merge back and convert to BGR
    lab_equalized = cv2.merge([l_equalized, a, b])
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)


# =============================================================================
# FULL PIPELINE — call this from detector / pipeline
# =============================================================================

def preprocess_frame(
    frame,
    resize: bool      = True,
    target_size: tuple = DEFAULT_SIZE,
    denoise_img: bool  = True,
    sharpen_img: bool  = True,
    equalize: bool     = True,
    normalize: bool    = False   # Keep False for YOLO (it normalizes internally)
) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a single frame.

    Steps (each can be toggled off):
        1. Equalize brightness (CLAHE)
        2. Denoise (bilateral filter)
        3. Sharpen (unsharp mask)
        4. Resize to YOLO input size
        5. Normalize pixel values (optional — YOLO handles this itself)

    Args:
        frame       : Raw BGR numpy array.
        resize      : Whether to resize. Default True.
        target_size : Target (W, H). Default (640, 640).
        denoise_img : Apply bilateral denoising. Default True.
        sharpen_img : Apply sharpening kernel. Default True.
        equalize    : Apply CLAHE brightness equalization. Default True.
        normalize   : Scale to [0,1]. Set True only for non-YOLO models.

    Returns:
        Preprocessed numpy array ready for inference.
    """
    processed = frame.copy()

    if equalize:
        processed = equalize_brightness(processed)

    if denoise_img:
        processed = denoise(processed)

    if sharpen_img:
        processed = sharpen(processed)

    if resize:
        processed = resize_image(processed, target_size)

    if normalize:
        processed = normalize_image(processed)

    return processed


# =============================================================================
# PLATE-SPECIFIC PREPROCESSING  (called AFTER detection, before OCR)
# =============================================================================

def preprocess_plate_crop(crop) -> np.ndarray:
    """
    Additional preprocessing applied specifically to a cropped plate region
    before OCR. Upscales and binarizes for maximum character clarity.

    Args:
        crop : Cropped BGR plate image (small region from detect_frame output).

    Returns:
        Grayscale thresholded image optimized for OCR.
    """
    # Upscale the small crop so OCR can read thin characters
    upscaled = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold: handles uneven lighting across the plate
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8
    )

    return binary


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python preprocessor.py path/to/image.jpg")
        raise SystemExit

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")

    result = preprocess_frame(img)
    print(f"[OK] Input shape : {img.shape}")
    print(f"[OK] Output shape: {result.shape}")

    cv2.imshow("Original",    img)
    cv2.imshow("Preprocessed", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()