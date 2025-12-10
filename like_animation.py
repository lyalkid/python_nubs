"""
Рука с вращением в локте.
От вертикального положения (вниз) до горизонтального (вправо).
"""

import numpy as np
from PIL import Image


# ==================== NURBS ====================

def basis_function(i, p, u, T):
    """B-сплайн базисная функция"""
    if p == 0:
        return 1.0 if T[i] <= u < T[i + 1] else 0.0
    
    left, right = 0.0, 0.0
    if T[i + p] - T[i] != 0:
        left = (u - T[i]) / (T[i + p] - T[i]) * basis_function(i, p - 1, u, T)
    if T[i + p + 1] - T[i + 1] != 0:
        right = (T[i + p + 1] - u) / (T[i + p + 1] - T[i + 1]) * basis_function(i + 1, p - 1, u, T)
    return left + right


def nurbs_curve(control_points, weights, degree=3, num_samples=100):
    """NURBS кривая"""
    n = len(control_points)
    knot_vector = [0] * (degree + 1) + list(range(1, n - degree)) + [n - degree] * (degree + 1)
    
    curve = []
    u_min, u_max = knot_vector[degree], knot_vector[-(degree+1)]
    for u in np.linspace(u_min, u_max, num_samples):
        if u >= u_max:
            u = u_max - 1e-10
        num_x, num_y, denom = 0.0, 0.0, 0.0
        for i in range(n):
            N = basis_function(i, degree, u, knot_vector)
            w = weights[i]
            num_x += N * w * control_points[i][0]
            num_y += N * w * control_points[i][1]
            denom += N * w
        if denom > 1e-10:
            curve.append([num_x / denom, num_y / denom])
    return curve


# ==================== БЕЗЬЕ ====================

def bezier_curve(control_points, num_samples=50):
    """Кривая Безье произвольного порядка"""
    n = len(control_points) - 1
    curve = []
    for t in np.linspace(0, 1, num_samples):
        pts = [np.array(p, dtype=float) for p in control_points]
        for r in range(1, n + 1):
            for i in range(n - r + 1):
                pts[i] = (1 - t) * pts[i] + t * pts[i + 1]
        curve.append(pts[0].tolist())
    return curve


# ==================== РИСОВАНИЕ ====================

def draw_line(canvas, x0, y0, x1, y1, color, width=1):
    """Брезенхем"""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    
    while True:
        for wx in range(-width//2, width//2 + 1):
            for wy in range(-width//2, width//2 + 1):
                px, py = x0 + wx, y0 + wy
                if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                    canvas[py, px] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def draw_polygon_fill(canvas, points, color):
    """Заливка полигона (scanline)"""
    if len(points) < 3:
        return
    y_min = int(max(0, min(p[1] for p in points)))
    y_max = int(min(canvas.shape[0] - 1, max(p[1] for p in points)))
    
    for y in range(y_min, y_max + 1):
        intersections = []
        for i in range(len(points)):
            p1, p2 = points[i], points[(i + 1) % len(points)]
            if p1[1] == p2[1]:
                continue
            if min(p1[1], p2[1]) <= y < max(p1[1], p2[1]):
                x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                intersections.append(x)
        intersections.sort()
        for i in range(0, len(intersections) - 1, 2):
            x_start = int(max(0, intersections[i]))
            x_end = int(min(canvas.shape[1] - 1, intersections[i + 1]))
            canvas[y, x_start:x_end + 1] = color


def draw_curve(canvas, points, color, width=2):
    """Рисует кривую по точкам"""
    for i in range(len(points) - 1):
        draw_line(canvas, points[i][0], points[i][1], 
                  points[i+1][0], points[i+1][1], color, width)


# ==================== ТРАНСФОРМАЦИИ ====================

def rotate_point(point, center, angle_rad):
    """Поворот точки вокруг центра"""
    px, py = point[0] - center[0], point[1] - center[1]
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    new_x = px * cos_a - py * sin_a
    new_y = px * sin_a + py * cos_a
    return [new_x + center[0], new_y + center[1]]


def rotate_points(points, center, angle_rad):
    """Поворот списка точек"""
    return [rotate_point(p, center, angle_rad) for p in points]


# ==================== РУКА ====================

def draw_hand(canvas, elbow_pos, angle_deg):
    """
    Рисует руку с заданным углом в локте.
    angle_deg: 0 = вниз, 90 = вправо
    """
    skin = np.array([220, 185, 155])
    outline = np.array([60, 45, 30])
    
    angle_rad = np.radians(angle_deg)
    
    # Все координаты задаём относительно локтя (рука вниз)
    # Потом повернём всю руку
    
    # ===== ПРЕДПЛЕЧЬЕ =====
    forearm_length = 180
    forearm_width_top = 32
    forearm_width_bottom = 25
    
    # Контрольные точки (локальные, рука вниз)
    forearm_left_local = [
        [-forearm_width_top, 0],
        [-forearm_width_top - 5, forearm_length * 0.3],
        [-forearm_width_bottom - 5, forearm_length * 0.6],
        [-forearm_width_bottom, forearm_length]
    ]
    
    forearm_right_local = [
        [forearm_width_top, 0],
        [forearm_width_top - 5, forearm_length * 0.3],
        [forearm_width_bottom - 5, forearm_length * 0.6],
        [forearm_width_bottom - 10, forearm_length]
    ]
    
    # Поворачиваем и смещаем
    forearm_left = rotate_points(forearm_left_local, [0, 0], angle_rad)
    forearm_left = [[p[0] + elbow_pos[0], p[1] + elbow_pos[1]] for p in forearm_left]
    
    forearm_right = rotate_points(forearm_right_local, [0, 0], angle_rad)
    forearm_right = [[p[0] + elbow_pos[0], p[1] + elbow_pos[1]] for p in forearm_right]
    
    # Кривые Безье
    left_curve = bezier_curve(forearm_left, 50)
    right_curve = bezier_curve(forearm_right, 50)
    
    # Рисуем предплечье
    forearm_poly = left_curve + list(reversed(right_curve))
    draw_polygon_fill(canvas, forearm_poly, skin)
    draw_curve(canvas, left_curve, outline, 2)
    draw_curve(canvas, right_curve, outline, 2)
    
    # ===== ЗАПЯСТЬЕ (конец предплечья) =====
    wrist_local = [0, forearm_length]
    wrist_pos = rotate_point(wrist_local, [0, 0], angle_rad)
    wrist_pos = [wrist_pos[0] + elbow_pos[0], wrist_pos[1] + elbow_pos[1]]
    
    # ===== ЛАДОНЬ (NURBS) =====
    palm_height = 70
    palm_width = 60
    
    # Локальные координаты ладони (относительно запястья)
    palm_local = [
        [-palm_width//2, 10],
        [-palm_width//2 - 5, 45],
        [-palm_width//2, palm_height - 10],
        [-palm_width//3, palm_height],
        [0, palm_height + 5],
        [palm_width//3, palm_height],
        [palm_width//2, palm_height - 10],
        [palm_width//2 + 8, 45],
        [palm_width//2 + 5, 10],
        [palm_width//4, 0],
        [0, -5],
        [-palm_width//4, 0],
    ]
    
    # Поворачиваем ладонь
    palm_rotated = rotate_points(palm_local, [0, 0], angle_rad)
    palm_points = [[p[0] + wrist_pos[0], p[1] + wrist_pos[1]] for p in palm_rotated]
    
    palm_weights = [1.2, 1.0, 1.3, 1.0, 1.1, 1.0, 1.3, 1.0, 1.2, 1.0, 1.0, 1.0]
    palm_closed = palm_points + palm_points[:3]
    weights_closed = palm_weights + palm_weights[:3]
    
    palm_curve = nurbs_curve(palm_closed, weights_closed, degree=3, num_samples=80)
    
    draw_polygon_fill(canvas, palm_curve, skin)
    draw_curve(canvas, palm_curve, outline, 2)
    
    # ===== 4 ПАЛЬЦА =====
    fingers = [
        (-22, 60, 13),   # Мизинец
        (-8, 72, 14),    # Безымянный
        (8, 76, 15),     # Средний
        (21, 68, 14),    # Указательный
    ]
    
    finger_base_y = palm_height - 5
    
    for fx_off, f_length, f_width in fingers:
        # Локальные координаты пальца
        seg1, seg2, seg3 = f_length * 0.4, f_length * 0.3, f_length * 0.3
        w1, w2, w3 = f_width, f_width * 0.85, f_width * 0.7
        
        left_local = [
            [fx_off - w1//2, finger_base_y],
            [fx_off - w1//2, finger_base_y + seg1],
            [fx_off - w2//2, finger_base_y + seg1 + seg2],
            [fx_off - w3//2, finger_base_y + seg1 + seg2 + seg3 * 0.7],
            [fx_off, finger_base_y + f_length]
        ]
        
        right_local = [
            [fx_off + w1//2, finger_base_y],
            [fx_off + w1//2, finger_base_y + seg1],
            [fx_off + w2//2, finger_base_y + seg1 + seg2],
            [fx_off + w3//2, finger_base_y + seg1 + seg2 + seg3 * 0.7],
            [fx_off, finger_base_y + f_length]
        ]
        
        # Поворачиваем
        left_rot = rotate_points(left_local, [0, 0], angle_rad)
        left_pts = [[p[0] + wrist_pos[0], p[1] + wrist_pos[1]] for p in left_rot]
        
        right_rot = rotate_points(right_local, [0, 0], angle_rad)
        right_pts = [[p[0] + wrist_pos[0], p[1] + wrist_pos[1]] for p in right_rot]
        
        left_c = bezier_curve(left_pts, 30)
        right_c = bezier_curve(right_pts, 30)
        
        finger_poly = left_c + list(reversed(right_c))
        draw_polygon_fill(canvas, finger_poly, skin)
        draw_curve(canvas, left_c, outline, 1)
        draw_curve(canvas, right_c, outline, 1)
        
        # Суставы
        for frac, w in [(seg1 / f_length, w1), ((seg1 + seg2) / f_length, w2)]:
            joint_y = finger_base_y + f_length * frac
            j_left = rotate_point([fx_off - w//2 + 2, joint_y], [0, 0], angle_rad)
            j_right = rotate_point([fx_off + w//2 - 2, joint_y], [0, 0], angle_rad)
            j_left = [j_left[0] + wrist_pos[0], j_left[1] + wrist_pos[1]]
            j_right = [j_right[0] + wrist_pos[0], j_right[1] + wrist_pos[1]]
            draw_line(canvas, j_left[0], j_left[1], j_right[0], j_right[1], outline, 1)
    
    # ===== БОЛЬШОЙ ПАЛЕЦ =====
    thumb_base_local = [palm_width//2, 25]
    thumb_length = 45
    
    thumb_left_local = [
        [thumb_base_local[0] - 3, thumb_base_local[1]],
        [thumb_base_local[0] + 2, thumb_base_local[1] + 22],
        [thumb_base_local[0] + 4, thumb_base_local[1] + 42],
        [thumb_base_local[0] + 2, thumb_base_local[1] + thumb_length]
    ]
    
    thumb_right_local = [
        [thumb_base_local[0] + 16, thumb_base_local[1] - 8],
        [thumb_base_local[0] + 20, thumb_base_local[1] + 15],
        [thumb_base_local[0] + 18, thumb_base_local[1] + 38],
        [thumb_base_local[0] + 2, thumb_base_local[1] + thumb_length]
    ]
    
    thumb_left_rot = rotate_points(thumb_left_local, [0, 0], angle_rad)
    thumb_left_pts = [[p[0] + wrist_pos[0], p[1] + wrist_pos[1]] for p in thumb_left_rot]
    
    thumb_right_rot = rotate_points(thumb_right_local, [0, 0], angle_rad)
    thumb_right_pts = [[p[0] + wrist_pos[0], p[1] + wrist_pos[1]] for p in thumb_right_rot]
    
    thumb_left_c = bezier_curve(thumb_left_pts, 25)
    thumb_right_c = bezier_curve(thumb_right_pts, 25)
    
    thumb_poly = thumb_left_c + list(reversed(thumb_right_c))
    draw_polygon_fill(canvas, thumb_poly, skin)
    draw_curve(canvas, thumb_left_c, outline, 1)
    draw_curve(canvas, thumb_right_c, outline, 1)
    
    # Сустав большого пальца
    j1 = rotate_point([thumb_base_local[0], thumb_base_local[1] + 28], [0, 0], angle_rad)
    j2 = rotate_point([thumb_base_local[0] + 16, thumb_base_local[1] + 22], [0, 0], angle_rad)
    j1 = [j1[0] + wrist_pos[0], j1[1] + wrist_pos[1]]
    j2 = [j2[0] + wrist_pos[0], j2[1] + wrist_pos[1]]
    draw_line(canvas, j1[0], j1[1], j2[0], j2[1], outline, 1)


def create_frame(angle_deg):
    """Создаёт один кадр с заданным углом"""
    width, height = 700, 550
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Локоть ближе к центру сверху
    elbow_pos = [400, 50]
    
    draw_hand(canvas, elbow_pos, angle_deg)
    
    return canvas


def create_animation():
    """Создаёт анимацию подъёма руки"""
    frames = []
    total_frames = 100
    
    print("Создание анимации движения локтя...")
    print("0° = вниз, -90° = влево")
    
    for i in range(total_frames):
        # Плавное движение (ease in-out)
        t = i / (total_frames - 1)
        if t < 0.5:
            ease_t = 2 * t * t
        else:
            ease_t = 1 - (-2 * t + 2) ** 2 / 2
        
        angle = -ease_t * 90  # От 0° до -90° (влево)
        
        frame = create_frame(angle)
        frames.append(Image.fromarray(frame))
        
        if (i + 1) % 25 == 0:
            print(f"Кадр {i + 1}/{total_frames}, угол: {angle:.1f}°")
    
    # Сохраняем GIF
    output_gif = './outputs/hand_elbow.gif'
    print(f"\nСохранение: {output_gif}")
    frames[0].save(output_gif, save_all=True, append_images=frames[1:],
                   duration=40, loop=0)
    
    # Сохраняем несколько кадров для просмотра
    frames[0].save('./outputs/hand_0deg.png')
    frames[49].save('./outputs/hand_45deg.png')
    frames[99].save('./outputs/hand_90deg.png')
    
    print("Готово!")
    return frames


if __name__ == "__main__":
    create_animation()