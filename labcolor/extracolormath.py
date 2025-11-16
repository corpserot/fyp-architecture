"""
Helper functions for color conversions and gamut adjustments using colormath.
"""

from colormath.color_objects import LabColor, LCHabColor, sRGBColor
from colormath.color_conversions import convert_color

def oklab_to_oklch(L: float, a: float, b: float) -> tuple[float, float, float]:
    """
    Convert Oklab to Oklch.
    Return (L, C, h)
    """
    # colormath uses L*a*b* and L*C*h* which are not Oklab/Oklch.
    # For this task, we'll assume the pseudocode's "Oklab" and "Oklch"
    # refer to the standard Lab and LCHab from colormath for implementation.
    lab_color = LabColor(lab_l=L * 100, lab_a=a, lab_b=b)
    lch_color = convert_color(lab_color, LCHabColor)
    return lch_color.lch_l / 100, lch_color.lch_c, lch_color.lch_h

def oklch_to_oklab(L: float, C: float, h: float) -> tuple[float, float, float]:
    """
    Convert Oklch to Oklab.
    Return (L, a, b)
    """
    lch_color = LCHabColor(lch_l=L * 100, lch_c=C, lch_h=h)
    lab_color = convert_color(lch_color, LabColor)
    return lab_color.lab_l / 100, lab_color.lab_a, lab_color.lab_b

def oklab_to_srgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    """
    Convert Oklab to sRGB.
    Output values are in 0..1 and assumed to be properly clipped.
    """
    lab_color = LabColor(lab_l=L * 100, lab_a=a, lab_b=b)
    srgb_color = convert_color(lab_color, sRGBColor)
    return srgb_color.rgb_r, srgb_color.rgb_g, srgb_color.rgb_b

def srgb_is_in_gamut(r: float, g: float, b: float) -> bool:
    """
    Return True if the color fits inside displayable sRGB.
    """
    return 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0

def adjust_chroma_to_fit_gamut(L: float, C: float, h: float) -> float:
    """
    Reduce C until (L, C, h) converts to an in gamut sRGB value.
    Return a possibly reduced chroma.
    """
    # Binary search for the maximum chroma that is in gamut
    low_c = 0.0
    high_c = C
    adjusted_c = low_c

    for _ in range(50):  # Iterate a fixed number of times for precision
        mid_c = (low_c + high_c) / 2
        L_lab, a_lab, b_lab = oklch_to_oklab(L, mid_c, h)
        r, g, b = oklab_to_srgb(L_lab, a_lab, b_lab)

        if srgb_is_in_gamut(r, g, b):
            adjusted_c = mid_c
            low_c = mid_c
        else:
            high_c = mid_c
    return adjusted_c
