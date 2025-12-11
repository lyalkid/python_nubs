"""
Пример отдельной отрисовки пальцев (без предплечья) для отладки сгиба.
"""

import os
import numpy as np
from PIL import Image

from like_animation import (
    bezier_curve,
    bend_finger_with_joints,
    thicken_finger,
    draw_polygon_fill,
    draw_curve,
    rotate_points,
)


def draw_fingers_only(canvas, origin, bend_factor=1.0, angle_deg=0.0):
    """Рисует ладонь и четыре пальца с заданным коэффициентом сгиба."""
    skin = np.array([220, 185, 155])
    outline = np.array([60, 45, 30])

    palm_height = 70
    palm_width = 60
    angle_rad = np.radians(angle_deg)

    palm_local = [
        [-palm_width // 2, 10],
        [-palm_width // 2 - 5, 45],
        [-palm_width // 2, palm_height - 10],
        [-palm_width // 3, palm_height],
        [-palm_width // 3, palm_height],
        [0, palm_height + 5],
        [palm_width // 3, palm_height],
        [palm_width // 2, palm_height - 10],
        [palm_width // 2 + 8, 45],
        [palm_width // 2 + 5, 10],
        [palm_width // 4, 0],
        [0, -5],
        [-palm_width // 4, 0],
    ]

    palm_rotated = rotate_points(palm_local, [0, 0], angle_rad)
    palm_points = [[p[0] + origin[0], p[1] + origin[1]] for p in palm_rotated]
    palm_poly = palm_points + palm_points[:1]
    draw_polygon_fill(canvas, palm_poly, skin)
    draw_curve(canvas, palm_points + palm_points[:2], outline, 2)

    fingers = [
        (-22, 60, 13),   # Мизинец
        (-8, 72, 14),    # Безымянный
        (8, 76, 15),     # Средний
        (21, 68, 14),    # Указательный
    ]

    finger_base_y = palm_height - 5
    finger_bend_rad = -np.clip(bend_factor, 0.0, 1.0) * np.radians(150)
    thicken_factor = 1.0 + np.clip(bend_factor, 0.0, 1.0) * 0.45

    for fx_off, f_length, f_width in fingers:
        seg1, seg2, seg3 = f_length * 0.4, f_length * 0.3, f_length * 0.3
        w1, w2, w3 = f_width, f_width * 0.85, f_width * 0.7
        joint1_y = finger_base_y + seg1
        joint2_y = finger_base_y + seg1 + seg2

        left_local = [
            [fx_off - w1 // 2, finger_base_y],
            [fx_off - w1 // 2, finger_base_y + seg1],
            [fx_off - w2 // 2, finger_base_y + seg1 + seg2],
            [fx_off - w3 // 2, finger_base_y + seg1 + seg2 + seg3 * 0.7],
            [fx_off, finger_base_y + f_length],
        ]

        right_local = [
            [fx_off + w1 // 2, finger_base_y],
            [fx_off + w1 // 2, finger_base_y + seg1],
            [fx_off + w2 // 2, finger_base_y + seg1 + seg2],
            [fx_off + w3 // 2, finger_base_y + seg1 + seg2 + seg3 * 0.7],
            [fx_off, finger_base_y + f_length],
        ]

        base_point = [fx_off, finger_base_y]
        left_bent = bend_finger_with_joints(left_local, base_point, joint1_y, joint2_y, finger_bend_rad)
        right_bent = bend_finger_with_joints(right_local, base_point, joint1_y, joint2_y, finger_bend_rad)
        left_bent = thicken_finger(left_bent, base_point[0], thicken_factor)
        right_bent = thicken_finger(right_bent, base_point[0], thicken_factor)

        left_rot = rotate_points(left_bent, [0, 0], angle_rad)
        right_rot = rotate_points(right_bent, [0, 0], angle_rad)
        left_pts = [[p[0] + origin[0], p[1] + origin[1]] for p in left_rot]
        right_pts = [[p[0] + origin[0], p[1] + origin[1]] for p in right_rot]

        left_c = bezier_curve(left_pts, 30)
        right_c = bezier_curve(right_pts, 30)

        finger_poly = left_c + list(reversed(right_c))
        draw_polygon_fill(canvas, finger_poly, skin)
        draw_curve(canvas, left_c, outline, 1)
        draw_curve(canvas, right_c, outline, 1)


def main():
    os.makedirs("outputs", exist_ok=True)
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    draw_fingers_only(canvas, origin=[150, 40], bend_factor=1.0, angle_deg=0.0)
    Image.fromarray(canvas).save("outputs/fingers_demo.png")
    print("Сохранено: outputs/fingers_demo.png")


if __name__ == "__main__":
    main()
