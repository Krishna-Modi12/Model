import numpy as np
import math

class SkinToneUtility:
    """
    Computes Individual Typology Angle (ITA) and mapping to 
    Fitzpatrick and Monk scales as per PRD Section 5.4.
    """
    
    @staticmethod
    def calculate_ita(lab_img, mask=None):
        """
        ITA = arctan2((L - 50), b) * (180 / pi)
        lab_img: cv2 LAB image
        mask: binary mask of skin region
        """
        if mask is not None:
            l_values = lab_img[:, :, 0][mask > 0]
            b_values = lab_img[:, :, 2][mask > 0]
        else:
            l_values = lab_img[:, :, 0]
            b_values = lab_img[:, :, 2]
            
        l_mean = np.mean(l_values)
        b_mean = np.mean(b_values)
        
        # Scale L from [0, 255] back to [0, 100] as per standard Lab
        # Scaled b should also be centered (-128 to 127)
        l_norm = (l_mean / 255.0) * 100.0
        b_norm = b_mean - 128.0
        
        ita = math.atan2(l_norm - 50, b_norm) * (180.0 / math.pi)
        return ita

    @staticmethod
    def get_scales(ita):
        """
        Approximate mapping of ITA to Fitzpatrick and Monk MST.
        Ref: Wilkes et al. (2015) and PRD Section 3.3.
        """
        # Fitzpatrick Mapping
        if ita > 55: fitz = 0    # Type I
        elif ita > 41: fitz = 1  # Type II
        elif ita > 28: fitz = 2  # Type III
        elif ita > 10: fitz = 3  # Type IV
        elif ita > -30: fitz = 4 # Type V
        else: fitz = 5           # Type VI
        
        # Monk Scale (MST 1-10) - Simplified Linear Mapping
        monk = int(np.clip((55 - ita) / 10 + 1, 1, 10))
        
        return fitz, monk
