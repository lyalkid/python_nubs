import numpy as np
from PIL import Image


def basis_function(i, p, u, T):
    """Вычисляет B-сплайн базисную функцию N{i,p}(u) используя формулу Бура"""
    if p == 0:
        return 1.0 if T[i] <= u < T[i + 1] else (1.0 if u == T[-1] and i == len(T) - 2 else 0.0)
    else:
        left = 0.0
        right = 0.0

        if T[i + p] - T[i] != 0:
            left = (u - T[i]) / (T[i + p] - T[i]) * basis_function(i, p - 1, u, T)

        if T[i + p + 1] - T[i + 1] != 0:
            right = (T[i + p + 1] - u) / (T[i + p + 1] - T[i + 1]) * basis_function(i + 1, p - 1, u, T)

        return left + right


def nurbs_curve(control_points, weights, knot_vector, degree, num_samples=100):
    """Вычисляет NURBS кривую"""
    curve_points = []

    # Проверяем корректность параметров
    n = len(control_points) - 1
    if len(knot_vector) != n + degree + 2:
        # Автоматически генерируем корректный узловой вектор
        knot_vector = list(range(n + degree + 2))

    for u in np.linspace(knot_vector[0], knot_vector[-1], num_samples):
        numerator = np.array([0.0, 0.0])
        denominator = 0.0

        for i in range(len(control_points)):
            N = basis_function(i, degree, u, knot_vector)
            numerator += weights[i] * N * np.array(control_points[i])
            denominator += weights[i] * N

        if denominator > 1e-10:
            curve_points.append(numerator / denominator)

    return np.array(curve_points)


def draw_line_bresenham(canvas, x0, y0, x1, y1, color, linewidth=1):
    """Рисует линию используя алгоритм Брезенхема"""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    points = []

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    # Рисуем линию с толщиной
    for px, py in points:
        for dx in range(-linewidth // 2, linewidth // 2 + 1):
            for dy in range(-linewidth // 2, linewidth // 2 + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < canvas.shape[1] and 0 <= ny < canvas.shape[0]:
                    canvas[ny, nx] = color


def draw_filled_polygon(canvas, points, color):
    """Закрашивает полигон используя алгоритм scanline"""
    if len(points) < 3:
        return

    # Находим границы полигона
    y_min = int(min(p[1] for p in points))
    y_max = int(max(p[1] for p in points))

    # Для каждой строки сканирования находим пересечения
    for y in range(y_min, y_max + 1):
        intersections = []

        # Проверяем каждое ребро
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            y1, y2 = p1[1], p2[1]
            x1, x2 = p1[0], p2[0]

            # Если ребро горизонтальное, пропускаем
            if y1 == y2:
                continue

            # Если y между y1 и y2
            if min(y1, y2) <= y < max(y1, y2):
                # Вычисляем x пересечения
                x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                intersections.append(x)

        # Сортируем пересечения
        intersections.sort()

        # Закрашиваем между парами пересечений
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = int(intersections[i])
                x_end = int(intersections[i + 1])
                for x in range(x_start, x_end + 1):
                    if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                        canvas[y, x] = color


def rotate_point(point, center, angle_deg):
    """Поворачивает точку вокруг центра на угол в градусах"""
    angle_rad = np.radians(angle_deg)
    px, py = point
    cx, cy = center

    # Смещаем точку в систему координат с центром в центре вращения
    px -= cx
    py -= cy

    # Поворачиваем
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    new_x = px * cos_a - py * sin_a
    new_y = px * sin_a + py * cos_a

    # Возвращаем в исходную систему координат
    return np.array([new_x + cx, new_y + cy])


def get_perpendicular_offset(point1, point2, thickness):
    """Возвращает перпендикулярное смещение для создания толщины"""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    length = np.sqrt(dx * dx + dy * dy)

    if length < 1e-6:
        return 0, 0

    # Нормализованный перпендикулярный вектор
    perp_x = -dy / length * thickness / 2
    perp_y = dx / length * thickness / 2

    return perp_x, perp_y


def draw_finger(canvas, start_point, length, angle, color, outline_color, is_thumb=False, curl=0.0, lift=0.0):
    """Рисует палец с контролем сгиба (curl) и подъема (lift)."""
    curl = np.clip(curl, 0.0, 1.0)
    lift = np.clip(lift, 0.0, 1.0)

    # Контрольные точки для пальца (B-сплайн)
    if is_thumb:
        finger_points = [
            [0, 0],
            [5, length * 0.25],
            [10, length * 0.45],
            [8, length * 0.7],
            [0, length],
        ]
        thickness_start = 18
        thickness_end = 8
        # Поднимаем большой палец вверх и назад к кисти
        finger_points[2][1] -= length * 0.35 * lift
        finger_points[3][1] -= length * 0.55 * lift
        finger_points[4][1] -= length * 0.7 * lift
        finger_points[3][0] -= 10 * lift
    else:
        finger_points = [
            [0, 0],
            [3, length * 0.3],
            [5, length * 0.6],
            [3, length * 0.9],
            [0, length],
        ]
        thickness_start = 15
        thickness_end = 6
        # Сгибаем палец в кулак
        finger_points[2][1] -= length * 0.35 * curl
        finger_points[3][1] -= length * 0.55 * curl
        finger_points[4][1] -= length * 0.7 * curl
        finger_points[3][0] -= 8 * curl

    # Поворачиваем и смещаем точки пальца
    control_points = []
    for point in finger_points:
        rotated = rotate_point(point, [0, 0], angle)
        control_points.append([rotated[0] + start_point[0],
                               rotated[1] + start_point[1]])

    # Создаем верхнюю и нижнюю кривые для пальца
    top_control_points = []
    bottom_control_points = []

    for i, point in enumerate(control_points):
        t = i / (len(control_points) - 1)
        thickness = thickness_start * (1 - t) + thickness_end * t

        if i == 0:
            if len(control_points) > 1:
                next_point = control_points[1]
                perp_x, perp_y = get_perpendicular_offset(point, next_point, thickness)
            else:
                perp_x, perp_y = thickness / 2, 0
        elif i == len(control_points) - 1:
            prev_point = control_points[i - 1]
            perp_x, perp_y = get_perpendicular_offset(prev_point, point, thickness)
        else:
            prev_point = control_points[i - 1]
            next_point = control_points[i + 1]
            perp1_x, perp1_y = get_perpendicular_offset(prev_point, point, thickness)
            perp2_x, perp2_y = get_perpendicular_offset(point, next_point, thickness)
            perp_x = (perp1_x + perp2_x) / 2
            perp_y = (perp1_y + perp2_y) / 2

        top_point = [point[0] + perp_x, point[1] + perp_y]
        bottom_point = [point[0] - perp_x, point[1] - perp_y]

        top_control_points.append(top_point)
        bottom_control_points.append(bottom_point)

    # Создаем кривые
    weights = [1.0] * len(control_points)
    knot_vector = list(range(len(control_points) + 3))
    degree = 2

    top_curve = nurbs_curve(top_control_points, weights, knot_vector, degree, num_samples=50)
    bottom_curve = nurbs_curve(bottom_control_points, weights, knot_vector, degree, num_samples=50)

    # Создаем полигон
    polygon_points = []
    for point in top_curve:
        polygon_points.append([point[0], point[1]])
    for point in reversed(bottom_curve):
        polygon_points.append([point[0], point[1]])

    # Закрашиваем палец
    draw_filled_polygon(canvas, polygon_points, color)

    # Рисуем контур
    for i in range(len(top_curve) - 1):
        draw_line_bresenham(canvas,
                            top_curve[i][0], top_curve[i][1],
                            top_curve[i + 1][0], top_curve[i + 1][1],
                            outline_color, linewidth=1)

    for i in range(len(bottom_curve) - 1):
        draw_line_bresenham(canvas,
                            bottom_curve[i][0], bottom_curve[i][1],
                            bottom_curve[i + 1][0], bottom_curve[i + 1][1],
                            outline_color, linewidth=1)


def create_soft_hand_shape(wrist_point, angle, hand_width=60, hand_height=40):
    """Создает мягкую овальную форму ладони"""
    # Более мягкая форма ладони - овал вместо прямоугольника
    hand_local_points = [
        [-hand_width * 0.35, -hand_height * 0.1],  # Верх-левый
        [-hand_width * 0.45, hand_height * 0.15],  # Середина-левый
        [-hand_width * 0.4, hand_height * 0.45],  # Низ-левый
        [-hand_width * 0.2, hand_height * 0.8],  # Низ-центр-левый
        [0, hand_height * 0.85],  # Самый низ (центр)
        [hand_width * 0.2, hand_height * 0.8],  # Низ-центр-правый
        [hand_width * 0.4, hand_height * 0.45],  # Низ-правый
        [hand_width * 0.45, hand_height * 0.15],  # Середина-правый
        [hand_width * 0.35, -hand_height * 0.1],  # Верх-правый
        [hand_width * 0.15, -hand_height * 0.2],  # Верх-центр-правый
        [0, -hand_height * 0.25],  # Самый верх (центр)
        [-hand_width * 0.15, -hand_height * 0.2]  # Верх-центр-левый
    ]

    # Поворачиваем и смещаем точки
    hand_points = []
    for point in hand_local_points:
        rotated = rotate_point(point, [0, 0], angle - 90)
        hand_points.append([rotated[0] + wrist_point[0],
                            rotated[1] + wrist_point[1]])

    return hand_points


def get_finger_y_offset(angle, finger_index, total_fingers=4):
    """Возвращает Y-смещение для пальца в зависимости от угла руки"""
    # Угол от -180 (вниз) до -90 (вправо)
    # Нормализуем угол от 0 до 1
    normalized_angle = (angle + 180) / 90  # от 0 до 1

    # Для каждого пальца разное смещение
    # Когда рука вниз (angle = -180): все пальцы на одном уровне (y = 0)
    # Когда рука вправо (angle = -90): пальцы на разных уровнях
    offsets = [0, 5, 10, 15]  # Смещения для 4 пальцев

    # Интерполируем в зависимости от угла
    current_offset = offsets[finger_index] * normalized_angle

    return current_offset

def create_forearm_frame(frame_num, total_frames=100):
    """Создает один кадр анимации предплечья с кистью"""
    width, height = 800, 600
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Цвета
    skin_color = np.array([200, 150, 120])
    outline_color = np.array([160, 120, 90])
    hand_color = np.array([190, 140, 110])

    # Центр вращения (локоть)
    elbow_center = np.array([width // 2 - 170, height // 2 - 30])

    # Угол от -180° (вертикально вниз) до -90° (горизонтально вправо)
    lift_progress = min(frame_num / (total_frames * 0.5 - 1), 1.0)
    angle = -180 + 90 * lift_progress  # -180 (вниз) -> -90 (вправо)

    # Управляющие точки для ЦЕНТРАЛЬНОЙ линии предплечья
    central_points = [
        [0, 0],
        [10, -40],
        [15, -80],
        [12, -120],
        [8, -160],
        [4, -200],
        [0, -250]
    ]

    # Поворачиваем центральные контрольные точки
    central_control_points = []
    for point in central_points:
        rotated = rotate_point(point, [0, 0], angle)
        central_control_points.append([rotated[0] + elbow_center[0],
                                       rotated[1] + elbow_center[1]])

    # Веса
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # Узловой вектор для B-сплайна степени 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]

    # Степень кривой
    degree = 2

    # Генерируем центральную кривую предплечья
    central_curve = nurbs_curve(central_control_points, weights, knot_vector, degree, num_samples=100)

    # Создаем ДВЕ параллельные кривыe для толщины предплечья
    top_control_points = []
    bottom_control_points = []

    for i, point in enumerate(central_control_points):
        t = i / (len(central_control_points) - 1)
        thickness = 55 * (1 - t) + 25 * t

        if i == 0:
            if len(central_control_points) > 1:
                next_point = central_control_points[1]
                perp_x, perp_y = get_perpendicular_offset(point, next_point, thickness)
            else:
                perp_x, perp_y = thickness / 2, 0
        elif i == len(central_control_points) - 1:
            prev_point = central_control_points[i - 1]
            perp_x, perp_y = get_perpendicular_offset(prev_point, point, thickness)
        else:
            prev_point = central_control_points[i - 1]
            next_point = central_control_points[i + 1]
            perp1_x, perp1_y = get_perpendicular_offset(prev_point, point, thickness)
            perp2_x, perp2_y = get_perpendicular_offset(point, next_point, thickness)
            perp_x = (perp1_x + perp2_x) / 2
            perp_y = (perp1_y + perp2_y) / 2

        top_point = [point[0] + perp_x, point[1] + perp_y]
        bottom_point = [point[0] - perp_x, point[1] - perp_y]

        top_control_points.append(top_point)
        bottom_control_points.append(bottom_point)

    # Генерируем верхнюю и нижнюю кривые предплечья
    top_curve = nurbs_curve(top_control_points, weights, knot_vector, degree, num_samples=100)
    bottom_curve = nurbs_curve(bottom_control_points, weights, knot_vector, degree, num_samples=100)

    # Создаем полигон для предплечья
    forearm_polygon = []
    for point in top_curve:
        forearm_polygon.append([point[0], point[1]])
    for point in reversed(bottom_curve):
        forearm_polygon.append([point[0], point[1]])

    # Закрашиваем предплечье
    draw_filled_polygon(canvas, forearm_polygon, skin_color)

    # Рисуем контуры предплечья
    for i in range(len(top_curve) - 1):
        draw_line_bresenham(canvas,
                            top_curve[i][0], top_curve[i][1],
                            top_curve[i + 1][0], top_curve[i + 1][1],
                            outline_color, linewidth=2)

    for i in range(len(bottom_curve) - 1):
        draw_line_bresenham(canvas,
                            bottom_curve[i][0], bottom_curve[i][1],
                            bottom_curve[i + 1][0], bottom_curve[i + 1][1],
                            outline_color, linewidth=2)

    # КИСТЬ (начинается от конца предплечья)
    wrist_point = central_curve[-1]

    # Создаем МЯГКУЮ форму ладони
    hand_points = create_soft_hand_shape(wrist_point, angle, hand_width=65, hand_height=45)

    # Закрашиваем ладонь
    draw_filled_polygon(canvas, hand_points, hand_color)

    # Рисуем контур ладони
    for i in range(len(hand_points)):
        p1 = hand_points[i]
        p2 = hand_points[(i + 1) % len(hand_points)]
        draw_line_bresenham(canvas,
                            p1[0], p1[1],
                            p2[0], p2[1],
                            outline_color, linewidth=2)

    # ПАЛЬЦЫ (4 пальца, прижаты друг к другу, смотрят вниз -> сгибаются в кулак)
    finger_lengths = [80, 85, 80, 70]
    finger_positions = [-22.5, -7.5, 7.5, 22.5]

    # Определяем точку начала пальцев (нижний центр кисти)
    # Находим две нижние точки ладони
    hand_bottom_points = []
    for point in hand_points:
        # Ищем точки с максимальной Y (самые нижние)
        if point[1] > wrist_point[1] + 20:  # Ниже запястья
            hand_bottom_points.append(point)

    if len(hand_bottom_points) >= 2:
        # Сортируем по X
        hand_bottom_points.sort(key=lambda p: p[0])
        # Центр нижней части
        bottom_left = hand_bottom_points[0]
        bottom_right = hand_bottom_points[-1]
        finger_base_center = [(bottom_left[0] + bottom_right[0]) / 2,
                              (bottom_left[1] + bottom_right[1]) / 2]
    else:
        # Запасной вариант
        finger_base_center = [wrist_point[0], wrist_point[1] + 45]

    curl_progress = np.clip((frame_num - total_frames * 0.3) / (total_frames * 0.7), 0, 1)
    for i, (pos, length) in enumerate(zip(finger_positions, finger_lengths)):
        finger_start_x = finger_base_center[0] + pos

        # Ключевое изменение: Y-координата зависит от угла и номера пальца
        # Когда рука вниз: все пальцы на одном уровне
        # Когда рука поднимается: пальцы смещаются по Y
        base_y = finger_base_center[1]

        # Сделай так:
        if angle <= -90:
            angle_progress = (angle + 180) / 90  # -180→0, -90→1
        else:
            angle_progress = 1.0  # Для angle = 90 тоже 1

        # Разное смещение для каждого пальца
        y_offsets = [0, 8, 16, 24]  # Указательный, средний, безымянный, мизинец
        y_offset = y_offsets[i] * angle_progress

        finger_start_y = base_y + y_offset  # ← ВАЖНО: добавляем смещение!

        # Угол пальцев: сгибаем вверх к кисти при curl_progress -> кулак
        finger_angle = angle + 180 - 110 * curl_progress

        # Рисуем палец
        draw_finger(canvas, [finger_start_x, finger_start_y], length * (1 - 0.25 * curl_progress),
                    finger_angle, skin_color, outline_color, is_thumb=False, curl=curl_progress)

    # БОЛЬШОЙ ПАЛЕЦ (отдельно, сбоку)
    thumb_length = 65
    # Большой палец начинается сбоку от кисти
    thumb_start_x = finger_base_center[0] + 35

    # Большой палец тоже смещается по Y
    thumb_lift = np.clip((frame_num - total_frames * 0.4) / (total_frames * 0.6), 0, 1)
    thumb_y_offset = -20 * angle_progress - 30 * thumb_lift  # Большой палец движется вверх
    thumb_start_y = finger_base_center[1] - 10 + thumb_y_offset

    # Угол большого пальца
    thumb_angle = angle - 15 + 180 - 170 * thumb_lift

    draw_finger(canvas, [thumb_start_x, thumb_start_y], thumb_length * (1 - 0.1 * thumb_lift),
                thumb_angle, skin_color, outline_color, is_thumb=True, lift=thumb_lift)

    return canvas


def create_forearm_animation():
    """Создает GIF анимацию движения предплечья с кистью"""
    frames = []
    total_frames = 100

    print("Создание анимации предплечья с кистью...")

    for frame_num in range(total_frames):
        # Создаем кадр
        frame = create_forearm_frame(frame_num, total_frames)

        # Преобразуем в PIL Image
        img = Image.fromarray(frame)
        frames.append(img)

        # Прогресс
        if (frame_num + 1) % 20 == 0:
            print(f"Создан кадр {frame_num + 1}/{total_frames}")

    # Сохраняем как GIF
    print("\nСохранение GIF...")
    frames[0].save(
        'forearm_with_hand.gif',
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
        optimize=True
    )

    print("Готово! Анимация сохранена как 'forearm_with_hand.gif'")


if __name__ == "__main__":
    create_forearm_animation()
