import cv2
import numpy as np
from PIL import Image
import os
import argparse

def order_points(pts):
    """Order points in the order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """Apply perspective transform to get bird's eye view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def detect_receipt_contour(image, debug=False):
    """Detect the receipt contour in the image using multiple methods"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
    combined = cv2.bitwise_or(edges, cv2.bitwise_or(thresh, adaptive_thresh))
    
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_area = image.shape[0] * image.shape[1]
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.02 or area > image_area * 0.95:
            continue
        
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
        
        aspect_ratio = max(width, height) / min(width, height)
        if 1.2 <= aspect_ratio <= 6.0:
            valid_contours.append((contour, area))
    
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    receipt_contour = None
    
    for contour, _ in valid_contours:
        for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08]:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                receipt_contour = approx
                break
        if receipt_contour is not None:
            break
    
    if receipt_contour is None and valid_contours:
        largest_contour = valid_contours[0][0]
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        receipt_contour = box.astype(int).reshape(-1, 1, 2)

    if receipt_contour is None and contours:
        largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        receipt_contour = box.astype(int).reshape(-1, 1, 2)

    return receipt_contour

def clean_receipt_image(image):
    """Clean the receipt image by applying denoising and thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    cleaned = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return cleaned

def resize_to_target_size(image, target_kb=200):
    """Resize image to approximately target KB size"""
    height, width = image.shape[:2]
    estimated_size_kb = (height * width * 3) / (1024 * 15)
    if estimated_size_kb <= target_kb:
        return image
    resize_factor = np.sqrt(target_kb / estimated_size_kb)
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    min_width, min_height = 800, 600
    if new_width < min_width or new_height < min_height:
        scale_w = min_width / width
        scale_h = min_height / height
        scale = min(scale_w, scale_h)
        new_width = int(width * scale)
        new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def save_as_pdf(image, output_path, target_kb=200):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    for quality in [95, 85, 75, 65, 55, 45, 35]:
        temp_path = "temp_receipt.jpg"
        pil_image.save(temp_path, "JPEG", quality=quality, optimize=True)
        file_size_kb = os.path.getsize(temp_path) / 1024
        if file_size_kb <= target_kb:
            img_for_pdf = Image.open(temp_path)
            img_for_pdf.save(output_path, "PDF", resolution=100.0)
            os.remove(temp_path)
            return
    pil_image.save(temp_path, "JPEG", quality=35, optimize=True)
    img_for_pdf = Image.open(temp_path)
    img_for_pdf.save(output_path, "PDF", resolution=100.0)
    os.remove(temp_path)

def process_receipt(input_path, output_jpg_path, output_pdf_path, debug=False):
    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not load image from {input_path}")
            return False
        
        print(f"Processing image: {input_path}")
        print(f"Original image size: {image.shape[1]}x{image.shape[0]}")
        
        receipt_contour = detect_receipt_contour(image, debug)
        
        if receipt_contour is not None:
            print("Receipt contour detected successfully")
            contour_area = cv2.contourArea(receipt_contour)
            image_area = image.shape[0] * image.shape[1]
            area_ratio = contour_area / image_area
            print(f"Contour area ratio: {area_ratio:.2f}")
            if area_ratio < 0.1:
                print("Warning: Detected contour too small, using original image")
                cleaned = clean_receipt_image(image)
            else:
                warped = four_point_transform(image, receipt_contour.reshape(4, 2))
                if warped.shape[0] < 100 or warped.shape[1] < 100:
                    print("Warning: Warped image too small, using original image")
                    cleaned = clean_receipt_image(image)
                else:
                    cleaned = clean_receipt_image(warped)
                    print(f"Cleaned image size: {cleaned.shape[1]}x{cleaned.shape[0]}")
        else:
            print("Warning: Could not detect receipt contour, using original image")
            cleaned = clean_receipt_image(image)
        
        resized = resize_to_target_size(cleaned, target_kb=200)
        print(f"Final image size: {resized.shape[1]}x{resized.shape[0]}")
        
        cv2.imwrite(output_jpg_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_size_kb = os.path.getsize(output_jpg_path) / 1024
        print(f"JPG file size: {jpg_size_kb:.1f} KB")
        if jpg_size_kb > 200:
            cv2.imwrite(output_jpg_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            jpg_size_kb = os.path.getsize(output_jpg_path) / 1024
            print(f"JPG file size after optimization: {jpg_size_kb:.1f} KB")
        
        save_as_pdf(resized, output_pdf_path, target_kb=200)
        pdf_size_kb = os.path.getsize(output_pdf_path) / 1024
        print(f"PDF file size: {pdf_size_kb:.1f} KB")
        print(f"Processing completed successfully!")
        print(f"Output files: {output_jpg_path}, {output_pdf_path}")
        return True
        
    except Exception as e:
        print(f"Error processing receipt: {str(e)}")
        try:
            original = cv2.imread(input_path)
            if original is not None:
                cv2.imwrite(output_jpg_path, original)
                save_as_pdf(original, output_pdf_path)
                print("Saved original image as fallback")
        except:
            pass
        return False

def main():
    parser = argparse.ArgumentParser(description='Process receipt image')
    parser.add_argument('--input', default='33.jpg', help='Input image path')
    parser.add_argument('--output-jpg', default='receipt_cleaned.jpg', help='Output JPG path')
    parser.add_argument('--output-pdf', default='receipt.pdf', help='Output PDF path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    success = process_receipt(args.input, args.output_jpg, args.output_pdf, args.debug)
    if success:
        print("Receipt processing completed successfully!")
    else:
        print("Receipt processing failed!")

if __name__ == "__main__":
    main()
