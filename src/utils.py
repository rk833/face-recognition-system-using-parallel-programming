# src/utils.py

import cv2 # type: ignore
import os
import time
import numpy as np # type: ignore
import face_recognition # type: ignore

def save_found_image(unknown_image, filename, output_folder, 
                     matched_face_location = None):
    """
    Draw bounding box only on the matched face and save the image.
    
    Args:
        unknown_image: The image in RGB format
        filename: Original filename
        output_folder: Where to save the result
        matched_face_location: (top, right, bottom, left) of the matched face only
    
    Returns:
        bool: True if saved successfully

    Explanation:
        The function draws a bounding box around the matched face in the image
    """
    try:
        # ensure output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # if no specific face location provided, draw on all faces (fallback)
        if matched_face_location is None:
            face_locations = face_recognition.face_locations(unknown_image)
            if not face_locations:
                print(f"   no faces detected in {filename} during save")
                return False
        else:
            # use only the matched face location
            face_locations = [matched_face_location]
        
        # draw bounding box only on the matched face(s)
        for top, right, bottom, left in face_locations:
            # draw green rectangle with thickness 3 for better visibility
            cv2.rectangle(
                unknown_image, 
                (left, top),      # top-left corner
                (right, bottom),  # bottom-right corner
                (0, 255, 0),      # green color in rgb
                3                 # line thickness
            )
            
            # add label "match" above the bounding box
            label = "MATCH"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            
            # get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # draw background rectangle for text
            cv2.rectangle(
                unknown_image,
                (left, top - text_height - 10),
                (left + text_width + 10, top),
                (0, 255, 0),  # green background
                -1  # filled rectangle
            )
            
            # draw text
            cv2.putText(
                unknown_image,
                label,
                (left + 5, top - 5),
                font,
                font_scale,
                (0, 0, 0),  # black text
                font_thickness
            )
        
        # convert rgb to bgr for opencv
        image_bgr = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)
        
        # create unique filename and save
        base_name = os.path.splitext(os.path.basename(filename))[0]
        timestamp = int(time.time())
        output_filename = f"detected_{base_name}_{timestamp}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        
        # save image
        success = cv2.imwrite(output_path, image_bgr)
        
        if success:
            print(f"   saved: {output_filename}")
        
        return success
        
    except Exception as e:
        print(f"  error saving {filename}: {str(e)}")
        return False


def validate_image_file(filepath):
    """
    Check whether a file is a valid image.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    # check extension
    _, ext = os.path.splitext(filepath.lower())
    if ext not in valid_extensions:
        return False
    
    # check if file exists and is readable
    if not os.path.isfile(filepath):
        return False
    
    # check file size (not empty)
    if os.path.getsize(filepath) == 0:
        return False
    
    return True


def get_image_files(folder_path):
    """
    Return valid image filenames from a folder.
    """
    if not os.path.exists(folder_path):
        return []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    image_files = []
    for entry in os.scandir(folder_path):
        if entry.is_file():
            _, ext = os.path.splitext(entry.name.lower())
            if ext in valid_extensions:
                image_files.append(entry.name)
    
    return image_files


def format_time(seconds):
    """
    Convert seconds into a readable time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        remainder = seconds % 3600
        minutes = int(remainder // 60)
        secs = remainder % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def calculate_speedup(serial_time, parallel_time, num_processes):
    """
    Compute speedup, efficiency, and overhead.
    """
    if parallel_time == 0:
        return {
            'speedup': 0,
            'efficiency': 0,
            'overhead': 0,
            'time_saved': 0
        }
    
    speedup = serial_time / parallel_time
    efficiency = (speedup / num_processes) * 100
    overhead = (parallel_time * num_processes) - serial_time
    time_saved = serial_time - parallel_time
    
    return {
        'speedup': speedup,
        'efficiency': efficiency,
        'overhead': overhead,
        'time_saved': time_saved
    }