"""Lightness units.

OpenCV's 8-bit LAB conversion stores CIELAB L* (0-100) scaled to 0-255. Every
darkness value in this project is therefore computed as (255 - L8) / L_SCALE,
which equals 100 - L*, and is reported in genuine CIELAB L* units. Thresholds
and colour-scale limits that were originally set in 8-bit units are divided by
the same factor, so all results are identical up to units.
"""
L_SCALE = 255.0 / 100.0   # 8-bit L per unit of CIELAB L*
