import cv2
import numpy as np
from models.image_format_info import ImageFormatInfo


def reconstruct_plane(data, offset, width, height, row_stride, pixel_stride):
    plane = np.frombuffer(data, dtype=np.uint8)
    output = np.empty((height, width), dtype=np.uint8)

    for row in range(height):
        start = offset + row * row_stride
        end = start + width * pixel_stride
        row_data = plane[start:end:pixel_stride]
        output[row, :] = row_data[:width]

    return output


def convert_yuv420_888_to_i420(
        raw_data: np.ndarray, 
        format_info: ImageFormatInfo, 
        uv_order="NV12"
    ):
    width = format_info.width
    height = format_info.height
    planes = format_info.planes

    if len(planes) != 3:
        raise ValueError("Expected 3 planes for YUV420_888 format")

    y_plane = reconstruct_plane(raw_data, 0, width, height, planes[0].row_stride, planes[0].pixel_stride)
    u_offset = planes[0].buffer_size
    chroma_width = width // 2
    chroma_height = height // 2
    pixel_stride_uv = planes[1].pixel_stride
    row_stride_uv = planes[1].row_stride

    if pixel_stride_uv == 1:
        u_plane = reconstruct_plane(raw_data, u_offset, chroma_width, chroma_height, row_stride_uv, 1)
        v_offset = u_offset + planes[1].buffer_size
        v_plane = reconstruct_plane(raw_data, v_offset, chroma_width, chroma_height, planes[2].row_stride, 1)
    else:
        uv_interleaved = np.frombuffer(raw_data, dtype=np.uint8)
        u_plane = np.empty((chroma_height, chroma_width), dtype=np.uint8)
        v_plane = np.empty((chroma_height, chroma_width), dtype=np.uint8)

        for row in range(chroma_height):
            row_start = u_offset + row * row_stride_uv
            row_data = uv_interleaved[row_start : row_start + chroma_width * pixel_stride_uv]
            if uv_order == "NV21":
                v_plane[row, :] = row_data[::2][:chroma_width]
                u_plane[row, :] = row_data[1::2][:chroma_width]
            else:
                u_plane[row, :] = row_data[::2][:chroma_width]
                v_plane[row, :] = row_data[1::2][:chroma_width]

    return np.concatenate([y_plane.ravel(), u_plane.ravel(), v_plane.ravel()])


def convert_yuv420_888_to_bgr(
        raw_data: np.ndarray, 
        format_info: ImageFormatInfo, 
        uv_order="NV12"
    ):
    height = format_info.height
    width = format_info.width

    i420 = convert_yuv420_888_to_i420(raw_data, format_info, uv_order).reshape((height * 3) // 2, width)
    bgr_img = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)

    return bgr_img


def measure_blur_laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def is_over_or_under_exposed(img_gray, low_thresh=0.02, high_thresh=0.02):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    cum = np.cumsum(hist)

    return cum[5] > low_thresh or cum[250] < high_thresh


def is_blur_image(img_bgr, blur_threshold=50.0):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = measure_blur_laplacian(img_gray)

    return blur_score < blur_threshold


def apply_clahe_tone_mapping(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) tone mapping.
    
    This is excellent for high dynamic range scenes with bright windows and dark interiors.
    
    Args:
        img_bgr: Input BGR image
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Tone-mapped BGR image
    """
    # Convert BGR to LAB color space (L=lightness, A/B=color channels)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel only (preserve color information)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_clahe = clahe.apply(l_channel)
    
    # Merge back and convert to BGR
    lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
    img_bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return img_bgr_clahe


def apply_gamma_correction(img_bgr, gamma=1.2):
    """
    Apply gamma correction to brighten dark areas.
    
    Args:
        img_bgr: Input BGR image
        gamma: Gamma value (>1 brightens, <1 darkens)
    
    Returns:
        Gamma-corrected BGR image
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    # Apply gamma correction
    return cv2.LUT(img_bgr, table)


def apply_tone_mapping(img_bgr, method="clahe", **kwargs):
    """
    Apply tone mapping to improve dynamic range.
    
    Args:
        img_bgr: Input BGR image
        method: "clahe", "gamma", or "clahe+gamma"
        **kwargs: Additional parameters for the specific method
    
    Returns:
        Tone-mapped BGR image
    """
    if method == "clahe":
        clip_limit = kwargs.get("clip_limit", 2.0)
        tile_grid_size = kwargs.get("tile_grid_size", (8, 8))
        return apply_clahe_tone_mapping(img_bgr, clip_limit, tile_grid_size)
    
    elif method == "gamma":
        gamma = kwargs.get("gamma", 1.2)
        return apply_gamma_correction(img_bgr, gamma)
    
    elif method == "clahe+gamma":
        # First apply CLAHE for local contrast
        clip_limit = kwargs.get("clip_limit", 2.0)
        tile_grid_size = kwargs.get("tile_grid_size", (8, 8))
        img_bgr = apply_clahe_tone_mapping(img_bgr, clip_limit, tile_grid_size)
        
        # Then apply mild gamma correction for overall brightness
        gamma = kwargs.get("gamma", 1.1)
        return apply_gamma_correction(img_bgr, gamma)
    
    else:
        raise ValueError(f"Unknown tone mapping method: {method}")