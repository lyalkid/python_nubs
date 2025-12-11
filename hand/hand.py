"""
Рука с вращением в локте и сгибанием пальцев в кулак.
"""

import numpy as np
from PIL import Image


# ==================== ЦВЕТА ====================

def shade(color, delta):
    """Смещает цвет на delta, оставаясь в границах 0-255."""
    return np.clip(color + delta, 0, 255)


SKIN_BASE = np.array([223, 190, 165])
OUTLINE_COLOR = np.array([75, 60, 45])

UPPER_ARM_COLOR = shade(SKIN_BASE, -2)
ELBOW_COLOR = shade(SKIN_BASE, -6)
FOREARM_COLOR = shade(SKIN_BASE, 4)
PALM_COLOR = shade(SKIN_BASE, 6)

THUMB_COLOR = shade(SKIN_BASE, -4)
PHALANX_SHADES = [0, -20, -40]  # для трёх фаланг

NAIL_COLOR = np.array([255, 4, 255])
NAIL_OUTLINE = np.array([170, 120, 110])


# ==================== NURBS ====================

def basis_function(i, p, u, T):
    if p == 0:
        return 1.0 if T[i] <= u < T[i + 1] else 0.0
    left, right = 0.0, 0.0
    if T[i + p] - T[i] != 0:
        left = (u - T[i]) / (T[i + p] - T[i]) * basis_function(i, p - 1, u, T)
    if T[i + p + 1] - T[i + 1] != 0:
        right = (T[i + p + 1] - u) / (T[i + p + 1] - T[i + 1]) * basis_function(i + 1, p - 1, u, T)
    return left + right


def nurbs_curve(control_points, weights, degree=3, num_samples=100):
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
    """Кривая Безье произвольного порядка (де Кастельжо)"""
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


def draw_joint_line(canvas, center, angle_rad, width, color, thickness=1):
    """Рисует поперечную линию сустава по центру сегмента."""
    perp = np.array([np.cos(angle_rad), -np.sin(angle_rad)])
    half = width / 2
    p1 = center - perp * half
    p2 = center + perp * half
    draw_line(canvas, p1[0], p1[1], p2[0], p2[1], color, thickness)


def draw_polygon_fill(canvas, points, color):
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
    for i in range(len(points) - 1):
        draw_line(canvas, points[i][0], points[i][1], 
                  points[i+1][0], points[i+1][1], color, width)


# ==================== ТРАНСФОРМАЦИИ ====================

def rotate_point(point, center, angle_rad):
    px, py = point[0] - center[0], point[1] - center[1]
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    new_x = px * cos_a - py * sin_a
    new_y = px * sin_a + py * cos_a
    return [new_x + center[0], new_y + center[1]]


def rotate_points(points, center, angle_rad):
    return [rotate_point(p, center, angle_rad) for p in points]


# ==================== ФАЛАНГА ====================

def draw_phalanx(canvas, start_pos, length, width_start, width_end, angle_rad, skin, outline,
                 is_tip=False, bend_amount=0, is_base=False, base_scale=0.5, round_tip=True):
    """
    Рисует одну фалангу пальца как кривую Безье.
    is_tip: если True - это последняя фаланга, рисуем ноготь
    Возвращает конечную позицию для следующей фаланги.
    """
    # Направление фаланги
    dir_vec = np.array([np.sin(angle_rad), np.cos(angle_rad)])
    perp_vec = np.array([dir_vec[1], -dir_vec[0]])
    
    # 4 угла фаланги
    p1 = np.array(start_pos) - perp_vec * width_start / 2
    p2 = np.array(start_pos) + perp_vec * width_start / 2
    end_pos = np.array(start_pos) + dir_vec * length
    p3 = end_pos + perp_vec * width_end / 2
    p4 = end_pos - perp_vec * width_end / 2
    
    # Контрольные точки для скругления (кривая Безье 3-го порядка)
    # Левая сторона
    left_cp1 = p1 + dir_vec * length * 0.3
    left_cp2 = p4 - dir_vec * length * 0.3
    left_bezier = [p1.tolist(), left_cp1.tolist(), left_cp2.tolist(), p4.tolist()]
    
    # Правая сторона
    right_cp1 = p2 + dir_vec * length * 0.3
    right_cp2 = p3 - dir_vec * length * 0.3
    right_bezier = [p2.tolist(), right_cp1.tolist(), right_cp2.tolist(), p3.tolist()]
    
    # Рисуем кривые
    left_curve = bezier_curve(left_bezier, 15)
    right_curve = bezier_curve(right_bezier, 15)
    
    # Закруглённый кончик для последней фаланги (опционально)
    tip_arc = []
    if is_tip and round_tip:
        radius = width_end / 2
        for t in np.linspace(-np.pi / 2, np.pi / 2, 12):
            pt = end_pos + dir_vec * (radius * np.cos(t)) + perp_vec * (radius * np.sin(t))
            tip_arc.append(pt.tolist())
    
    # Закругление у основания (чтобы палец «врастал» плавно)
    base_arc = []
    if is_base:
        radius = width_start * base_scale
        base_center = np.array(start_pos) - dir_vec * radius
        for t in np.linspace(np.pi / 2, -np.pi / 2, 10):
            pt = base_center + perp_vec * (radius * np.cos(t)) + dir_vec * (radius * np.sin(t))
            base_arc.append(pt.tolist())
    
    # Полигон фаланги (с арками на кончике и основании, если нужно)
    phalanx_poly = []
    if base_arc:
        phalanx_poly += base_arc
        phalanx_poly += left_curve[1:]  # продолжение без стартовой точки
    else:
        phalanx_poly += left_curve
    if is_tip and tip_arc:
        phalanx_poly += tip_arc
    phalanx_poly += list(reversed(right_curve))
    
    draw_polygon_fill(canvas, phalanx_poly, skin)
    draw_curve(canvas, left_curve, outline, 1)
    draw_curve(canvas, right_curve, outline, 1)
    
    # Ноготь на кончике пальца (виден при сгибании > 0.35)
    if is_tip and bend_amount > 0.35:
        nail_length = length * 0.7
        nail_width = width_start * 0.8
        
        nail_start = np.array(start_pos) + dir_vec * length * 0.2
        nail_end = nail_start + dir_vec * nail_length
        
        n1 = nail_start - perp_vec * nail_width / 2
        n2 = nail_start + perp_vec * nail_width / 2
        n3 = nail_end + perp_vec * nail_width / 3
        n4 = nail_end - perp_vec * nail_width / 3
        
        nail_tip = nail_end + dir_vec * nail_width * 0.15
        
        nail_points = [
            n1.tolist(),
            n4.tolist(),
            nail_tip.tolist(),
            n3.tolist(),
            n2.tolist(),
        ]
        
        nail_curve = bezier_curve(nail_points, 25)
        draw_polygon_fill(canvas, nail_curve, NAIL_COLOR)
        draw_curve(canvas, nail_curve, NAIL_OUTLINE, 1)
    
    return end_pos.tolist()


# ==================== ПАЛЕЦ ИЗ ФАЛАНГ ====================

def draw_finger(canvas, base_pos, finger_lengths, finger_widths, 
                arm_angle_rad, bend_amount, skin, outline, finger_index=0):
    """
    Рисует палец из 3 фаланг, сгибающийся ВНУТРЬ ладони (к зрителю).
    finger_index: номер пальца для разного цвета
    """
    base_skin = skin if skin is not None else SKIN_BASE
    phalanx_colors = [shade(base_skin, d) for d in PHALANX_SHADES]
    
    # Углы сгиба каждой фаланги (отрицательные - внутрь ладони)
    # Смягчённые углы, чтобы кончики не задирались вверх и ногти оставались видны
    bend1 = -bend_amount * np.pi * 0.02
    bend2 = -bend_amount * np.pi * 0.05
    bend3 = -bend_amount * np.pi * 0.02
    
    # Укорочение при сгибе к зрителю
    shorten1 = 1 - bend_amount * 0.2
    shorten2 = 1 - bend_amount * 0.35
    shorten3 = 1 - bend_amount * 0.5
    
    current_angle = arm_angle_rad
    current_pos = list(base_pos)
    
    # Фаланга 1 (проксимальная) - голубая
    current_pos = draw_phalanx(
        canvas, current_pos,
        finger_lengths[0] * shorten1,
        finger_widths[0], finger_widths[1],
        current_angle, phalanx_colors[0], outline,
        is_tip=False, bend_amount=bend_amount, is_base=True, base_scale=0.15
    )
    # линия сустава между 1 и 2 фалангами
    joint_width1 = (finger_widths[1] + finger_widths[0]) * 0.5
    draw_joint_line(canvas, np.array(current_pos), current_angle, joint_width1, outline, thickness=1)
    current_angle += bend1
    
    # Фаланга 2 (средняя) - зелёная
    current_pos = draw_phalanx(
        canvas, current_pos,
        finger_lengths[1] * shorten2,
        finger_widths[1], finger_widths[2],
        current_angle, phalanx_colors[1], outline,
        is_tip=False, bend_amount=bend_amount
    )
    # линия сустава между 2 и 3 фалангами
    joint_width2 = (finger_widths[2] + finger_widths[1]) * 0.5
    draw_joint_line(canvas, np.array(current_pos), current_angle, joint_width2, outline, thickness=1)
    current_angle += bend2
    
    # Фаланга 3 (дистальная) - персиковая, с ногтем
    current_pos = draw_phalanx(
        canvas, current_pos,
        finger_lengths[2] * shorten3,
        finger_widths[2], finger_widths[3],
        current_angle, phalanx_colors[2], outline,
        is_tip=True, bend_amount=bend_amount, round_tip=False
    )


def draw_thumb(canvas, base_pos, thumb_lengths, thumb_widths,
               arm_angle_rad, bend_amount, skin, outline):
    """Большой палец из 3 фаланг с мягким естественным тоном."""
    thumb_lift = -bend_amount * -3.8  # Против часовой, но без перекрута
    base_angle = arm_angle_rad + 0.25 + thumb_lift
    
    current_angle = base_angle
    current_pos = list(base_pos)
    bend_phalanx = bend_amount * 0.15
    
    base_skin = skin if skin is not None else np.array([223, 190, 165])
    thumb_color = shade(base_skin, -4)
    
    # Фаланга 1 (проксимальная)
    current_pos = draw_phalanx(
        canvas, current_pos,
        thumb_lengths[0],
        thumb_widths[0], thumb_widths[1],
        current_angle, thumb_color, outline,
        is_tip=False, bend_amount=bend_amount, is_base=True, base_scale=0.5
    )
    draw_joint_line(canvas, np.array(current_pos), current_angle, (thumb_widths[0] + thumb_widths[1]) * 0.5, outline, thickness=1)
    current_angle += bend_phalanx * 0.6
    
    # Фаланга 2 (средняя)
    current_pos = draw_phalanx(
        canvas, current_pos,
        thumb_lengths[1],
        thumb_widths[1], thumb_widths[2],
        current_angle, thumb_color, outline,
        is_tip=False, bend_amount=bend_amount
    )
    draw_joint_line(canvas, np.array(current_pos), current_angle, (thumb_widths[1] + thumb_widths[2]) * 0.5, outline, thickness=1)
    current_angle += bend_phalanx
    
    # Фаланга 3 (дистальная) - с ногтем
    draw_phalanx(
        canvas, current_pos,
        thumb_lengths[2],
        thumb_widths[2], thumb_widths[3],
        current_angle, thumb_color, outline,
        is_tip=True, bend_amount=bend_amount, round_tip=False
    )


# ==================== РУКА ====================

def draw_hand(canvas, elbow_pos, angle_deg, bend_amount=0):
    """
    Рисует руку с плечом.
    angle_deg: угол в локте (0 = вниз, -90 = влево)
    bend_amount: сгибание пальцев (0 = прямые, 1 = кулак)
    """
    outline = OUTLINE_COLOR
    skin_base = SKIN_BASE

    upper_arm_color = UPPER_ARM_COLOR
    elbow_color = ELBOW_COLOR
    forearm_color = FOREARM_COLOR
    palm_color = PALM_COLOR
    
    angle_rad = np.radians(angle_deg)
    
    # ===== ВЕРХНЯЯ ЧАСТЬ РУКИ (плечо -> локоть) =====
    # Плечо НЕПОДВИЖНО - всегда идёт вверх  
    upper_arm_length = 160
    upper_arm_width_top = 45  # У плеча толще
    upper_arm_width_bottom = 35  # У локтя тоньше
    
    # Плечо фиксировано - идёт вверх (угол 0)
    upper_angle = 0  # Всегда вертикально вверх
    
    upper_left_local = [
        [-upper_arm_width_bottom, 0],
        [-upper_arm_width_bottom - 3, -upper_arm_length * 0.3],
        [-upper_arm_width_top - 5, -upper_arm_length * 0.7],
        [-upper_arm_width_top, -upper_arm_length]
    ]
    
    upper_right_local = [
        [upper_arm_width_bottom, 0],
        [upper_arm_width_bottom + 5, -upper_arm_length * 0.3],
        [upper_arm_width_top + 8, -upper_arm_length * 0.7],
        [upper_arm_width_top + 5, -upper_arm_length]
    ]
    
    # Поворачиваем верхнюю часть руки
    upper_left = rotate_points(upper_left_local, [0, 0], upper_angle)
    upper_left = [[p[0] + elbow_pos[0], p[1] + elbow_pos[1]] for p in upper_left]
    
    upper_right = rotate_points(upper_right_local, [0, 0], upper_angle)
    upper_right = [[p[0] + elbow_pos[0], p[1] + elbow_pos[1]] for p in upper_right]
    
    upper_left_curve = bezier_curve(upper_left, 40)
    upper_right_curve = bezier_curve(upper_right, 40)
    
    upper_poly = upper_left_curve + list(reversed(upper_right_curve))
    draw_polygon_fill(canvas, upper_poly, upper_arm_color)  # Розовый
    draw_curve(canvas, upper_left_curve, outline, 2)
    draw_curve(canvas, upper_right_curve, outline, 2)
    
    # ===== ЛОКОТЬ (скругление/соединение) =====
    # Рисуем заполняющий эллипс в месте соединения
    elbow_fill = []
    for i in range(30):
        angle_e = i / 29 * 2 * np.pi
        x = elbow_pos[0] + np.cos(angle_e) * 38
        y = elbow_pos[1] + np.sin(angle_e) * 30
        elbow_fill.append([x, y])
    draw_polygon_fill(canvas, elbow_fill, elbow_color)  # Красный
    
    # ===== ПРЕДПЛЕЧЬЕ (Безье) =====
    forearm_length = 180
    forearm_width_top = 32
    forearm_width_bottom = 25
    
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
    
    forearm_left = rotate_points(forearm_left_local, [0, 0], angle_rad)
    forearm_left = [[p[0] + elbow_pos[0], p[1] + elbow_pos[1]] for p in forearm_left]
    
    forearm_right = rotate_points(forearm_right_local, [0, 0], angle_rad)
    forearm_right = [[p[0] + elbow_pos[0], p[1] + elbow_pos[1]] for p in forearm_right]
    
    left_curve = bezier_curve(forearm_left, 50)
    right_curve = bezier_curve(forearm_right, 50)
    
    forearm_poly = left_curve + list(reversed(right_curve))
    draw_polygon_fill(canvas, forearm_poly, forearm_color)  # Фиолетовый
    draw_curve(canvas, left_curve, outline, 2)
    draw_curve(canvas, right_curve, outline, 2)
    
    # ===== ЗАПЯСТЬЕ =====
    wrist_local = [0, forearm_length]
    wrist_pos = rotate_point(wrist_local, [0, 0], angle_rad)
    wrist_pos = [wrist_pos[0] + elbow_pos[0], wrist_pos[1] + elbow_pos[1]]
    
    # ===== ЛАДОНЬ (NURBS) =====
    palm_height = 70
    palm_width = 60
    
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
    
    palm_rotated = rotate_points(palm_local, [0, 0], angle_rad)
    palm_points = [[p[0] + wrist_pos[0], p[1] + wrist_pos[1]] for p in palm_rotated]
    
    palm_weights = [1.2, 1.0, 1.3, 1.0, 1.1, 1.0, 1.3, 1.0, 1.2, 1.0, 1.0, 1.0]
    palm_closed = palm_points + palm_points[:3]
    weights_closed = palm_weights + palm_weights[:3]
    
    palm_curve = nurbs_curve(palm_closed, weights_closed, degree=3, num_samples=80)
    
    draw_polygon_fill(canvas, palm_curve, palm_color)  # Жёлтый
    draw_curve(canvas, palm_curve, outline, 2)
    
    # ===== 4 ПАЛЬЦА ИЗ ФАЛАНГ =====
    # Каждый палец: (смещение_x, [длины фаланг], [ширины фаланг])
    fingers_data = [
        # Мизинец - самый короткий
        (-22, [22, 16, 14], [13, 11, 9, 5]),
        # Безымянный
        (-8, [28, 20, 16], [14, 12, 10, 5]),
        # Средний - самый длинный
        (8, [30, 22, 18], [15, 13, 11, 6]),
        # Указательный
        (22, [26, 20, 16], [14, 12, 10, 5]),
    ]
    
    finger_base_y = palm_height - 5
    
    for fx_off, f_lengths, f_widths in fingers_data:
        base_local = [fx_off, finger_base_y]
        base_rotated = rotate_point(base_local, [0, 0], angle_rad)
        base_world = [base_rotated[0] + wrist_pos[0], base_rotated[1] + wrist_pos[1]]
        
        draw_finger(canvas, base_world, f_lengths, f_widths,
                   angle_rad, bend_amount, skin_base, outline)
    
    # ===== БОЛЬШОЙ ПАЛЕЦ (2 фаланги) =====
    # Пододвигаем основание ближе к ладони
    thumb_base_local = [palm_width//2 + 2, 22]
    thumb_base_rotated = rotate_point(thumb_base_local, [0, 0], angle_rad)
    thumb_base = [thumb_base_rotated[0] + wrist_pos[0], thumb_base_rotated[1] + wrist_pos[1]]
    
    # Большой палец: 3 фаланги
    thumb_lengths = [24, 18, 14]
    thumb_widths = [18, 15, 12, 9]  # Начало -> кончик
    
    draw_thumb(canvas, thumb_base, thumb_lengths, thumb_widths,
               angle_rad, bend_amount, skin_base, outline)


def create_frame(angle_deg, bend_amount):
    """Создаёт один кадр"""
    width, height = 800, 700
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Локоть в центре по вертикали
    elbow_pos = [450, 350]
    draw_hand(canvas, elbow_pos, angle_deg, bend_amount)
    
    return canvas


def create_animation():
    """Создаёт анимацию: подъём руки + сжатие в кулак"""
    frames = []
    total_frames = 100
    
    print("Создание анимации...")
    print("Рука поднимается и пальцы сжимаются в кулак")
    
    for i in range(total_frames):
        t = i / (total_frames - 1)
        
        # Ease in-out
        if t < 0.5:
            ease_t = 2 * t * t
        else:
            ease_t = 1 - (-2 * t + 2) ** 2 / 2
        
        angle = -ease_t * 90  # 0 -> -90 (вниз -> влево)
        bend = ease_t         # 0 -> 1 (прямые -> кулак)
        
        frame = create_frame(angle, bend)
        frames.append(Image.fromarray(frame))
        
        if (i + 1) % 25 == 0:
            print(f"Кадр {i + 1}/{total_frames}, угол: {angle:.1f}°, сжатие: {bend:.2f}")
    
    # Сохраняем GIF
    output_gif = './outputs/hand_fist.gif'
    print(f"\nСохранение: {output_gif}")
    frames[0].save(output_gif, save_all=True, append_images=frames[1:],
                   duration=40, loop=0)
    
    # Ключевые кадры
    frames[0].save('./outputs/fist_0.png')
    frames[49].save('./outputs/fist_50.png')
    frames[99].save('./outputs/fist_100.png')
    
    print("Готово!")
    return frames


if __name__ == "__main__":
    create_animation()
