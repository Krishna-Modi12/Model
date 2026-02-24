import numpy as np

class SkinToneResult:
    def __init__(self, ita_value, fitzpatrick_type, monk_scale):
        self.ita_value = ita_value
        self.fitzpatrick_type = fitzpatrick_type
        self.monk_scale = monk_scale

class SkinToneAnalyzer:
    def __init__(self):
        pass

    def analyze(self, skin_pixels_lab):
        """
        Analyzes a list of LAB pixels to extract skin tone metrics.
        
        Args:
            skin_pixels_lab (numpy.ndarray): Array of shape (N, 3) containing LAB values.
                                              L is 0-255, A and B are 0-255 (OpenCV format).

        Returns:
            SkinToneResult or None if no pixels provided.
        """
        if skin_pixels_lab is None or len(skin_pixels_lab) == 0:
            return None
        
        # Calculate mean L*, a*, b*
        mean_lab = np.mean(skin_pixels_lab, axis=0)
        L_cv, a_cv, b_cv = mean_lab
        
        # Convert OpenCV LAB to standard CIELAB
        # OpenCV: L is 0-255 (represents 0-100), A is 0-255 (represents -128 to 127), B is 0-255
        L = L_cv * 100 / 255.0
        b = b_cv - 128.0
        
        # Compute ITA (Individual Typology Angle)
        # ITA = arctan((L - 50) / b) * (180 / pi)
        if b == 0:
            b = 1e-5 # avoid division by zero
            
        ita = np.arctan((L - 50.0) / b) * (180.0 / np.pi)
        
        # Determine Fitzpatrick Scale (I-VI) based on ITA
        if ita > 55:
            fitz = 1
        elif ita > 41:
            fitz = 2
        elif ita > 28:
            fitz = 3
        elif ita > 10:
            fitz = 4
        elif ita > -30:
            fitz = 5
        else:
            fitz = 6
            
        # Determine Monk Scale (1-10) heuristically based on ITA ranges
        # Monk 1 is lightest, Monk 10 is darkest.
        if ita > 60: monk = 1
        elif ita > 50: monk = 2
        elif ita > 40: monk = 3
        elif ita > 30: monk = 4
        elif ita > 20: monk = 5
        elif ita > 10: monk = 6
        elif ita > 0:  monk = 7
        elif ita > -15: monk = 8
        elif ita > -35: monk = 9
        else: monk = 10
            
        return SkinToneResult(ita_value=round(float(ita), 2), fitzpatrick_type=fitz, monk_scale=monk)
