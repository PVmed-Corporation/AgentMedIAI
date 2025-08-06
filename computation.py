import pandas as pd
import math
import numpy as np
from scipy.spatial.distance import cdist

def read_spacing(input_csv, input_name):
    df = pd.read_csv(f"{input_csv}")
    row_idx = "OriginalImagePixelSpacing"
    col_idx = f"{input_name}"
    spacing = df.at[row_idx, col_idx]
    spacing = [spacing, spacing]
    return spacing

def distance(coordinates1, coordinates2, spacing, unit="mm"):
    x1, y1 = coordinates1
    x2, y2 = coordinates2
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)** 2)*spacing
    return f"{distance}{unit}"

def area(label, mask_path, spacing, unit):
    physical_area = 0
    for i in label:
        # print(i)
        mask = np.load(mask_path)
        imgray = (mask[i,:,:]*255).astype(np.uint8)       
        pixel_count = np.sum(imgray)

        x_spacing, y_spacing = spacing
        single_pixel_area = x_spacing * y_spacing  
        physical_areai = pixel_count * single_pixel_area
        physical_area += physical_areai
    return f"{physical_area}{unit}"
    
def calculate_max_diameter(label, mask_path, pixel_spacing):
    for i in label:
        mask = np.load(mask_path)
        imgray = (mask[i,:,:]*255).astype(np.uint8)  
        coords = np.argwhere(imgray == 1)        
        if len(coords) < 2:        
            return 0.0, "pixel" if pixel_spacing is None else "mm"           
        distances = cdist(coords, coords, metric='euclidean')         
        max_pixel_distance = np.max(distances[distances > 0])            
                    
        avg_spacing = (pixel_spacing[0] + pixel_spacing[1]) / 2        
        max_diameteri = max_pixel_distance * avg_spacing
        max_diameter += max_diameteri
           
    return max_diameter

def percentage_change(current, previous):
    if previous == 0:
        return float('inf') if current > 0 else 0
    return ((current - previous) / previous) * 100