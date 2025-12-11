import numpy as np
import matplotlib.pyplot as plt
import random

# Параметры холста
canvas_width, canvas_height = 1920, 1080

# Цвета
canvas_color = (30/255, 33/255, 36/255)        # почти черный
moon_color = (255/255, 253/255, 209/255)       # бледно-желтый
crater_color = (180/255, 180/255, 180/255)     # светло-серый
pumpkin_color = (255/255, 140/255, 0/255)      # оранжевый
stem_color = (34/255, 102/255, 17/255)         # темно-зеленый
branch_color = (101/255, 67/255, 33/255)       # коричневый
raven_color = (20/255, 20/255, 20/255)         # очень черный
eye_color = (255/255, 220/255, 0/255)          # желтый
beak_color = (80/255, 80/255, 80/255)          # серый

# === СОБСТВЕННАЯ РЕАЛИЗАЦИЯ NURBS ===
def basis_function(i, p, u, T):
    """Вычисляет B-сплайн базисную функцию N_{i,p}(u) используя de Boor"""
    if p == 0:
        return 1.0 if T[i] <= u < T[i+1] else (1.0 if u == T[-1] and i == len(T) - 2 else 0.0)
    else:
        left = 0.0
        right = 0.0
        
        if T[i+p] - T[i] != 0:
            left = (u - T[i]) / (T[i+p] - T[i]) * basis_function(i, p-1, u, T)
        
        if T[i+p+1] - T[i+1] != 0:
            right = (T[i+p+1] - u) / (T[i+p+1] - T[i+1]) * basis_function(i+1, p-1, u, T)
        
        return left + right

def nurbs_curve(control_points, weights, knot_vector, degree, num_samples=100):
    """Вычисляет NURBS кривую"""
    curve_points = []
    
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

def bezier_curve(control_points, num_samples=100):
    """Вычисляет кривую Безье произвольного порядка"""
    curve_points = []
    degree = len(control_points) - 1
    
    for t in np.linspace(0, 1, num_samples):
        point = np.array([0.0, 0.0])
        
        for i, cp in enumerate(control_points):
            C = 1
            for k in range(i):
                C = C * (degree - k) // (k + 1)
            
            B = C * (t ** i) * ((1 - t) ** (degree - i))
            point += B * np.array(cp)
        
        curve_points.append(point)
    
    return np.array(curve_points)

def fill_polygon_scanline(canvas, points, color):
    """Заполняет многоугольник используя алгоритм сканирующей линии"""
    if len(points) < 3:
        return
    
    points = np.array(points, dtype=int)
    min_y = max(0, np.min(points[:, 1]))
    max_y = min(canvas.shape[0] - 1, np.max(points[:, 1]))
    
    for y in range(min_y, max_y + 1):
        intersections = []
        
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            y1, y2 = p1[1], p2[1]
            x1, x2 = p1[0], p2[0]
            
            if (y1 <= y < y2) or (y2 <= y < y1):
                if y1 != y2:
                    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(int(x))
        
        if len(intersections) >= 2:
            intersections.sort()
            for j in range(0, len(intersections) - 1, 2):
                x_start = max(0, intersections[j])
                x_end = min(canvas.shape[1] - 1, intersections[j + 1])
                canvas[y, x_start:x_end + 1] = color

def fill_circle_scanline(canvas, center_x, center_y, radius, color):
    """Заполняет круг используя алгоритм сканирующей линии"""
    center_x, center_y = int(center_x), int(center_y)
    radius = int(radius)
    
    for y in range(max(0, center_y - radius), min(canvas.shape[0], center_y + radius + 1)):
        dy = y - center_y
        if abs(dy) <= radius:
            dx = int(np.sqrt(radius**2 - dy**2))
            x_start = max(0, center_x - dx)
            x_end = min(canvas.shape[1] - 1, center_x + dx)
            canvas[y, x_start:x_end + 1] = color

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
        for dx in range(-linewidth//2, linewidth//2 + 1):
            for dy in range(-linewidth//2, linewidth//2 + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < canvas.shape[1] and 0 <= ny < canvas.shape[0]:
                    canvas[ny, nx] = color

def draw_bezier_with_line(canvas, points, color, linewidth=1):
    """Рисует кривую Безье линиями между точками"""
    for i in range(len(points) - 1):
        draw_line_bresenham(canvas, points[i][0], points[i][1], 
                           points[i+1][0], points[i+1][1], color, linewidth)

# Создаем холст (numpy array)
canvas = np.zeros((canvas_height, canvas_width, 3))
# Заполняем фон
for c in range(3):
    canvas[:, :, c] = canvas_color[c]

# === Параметры луны для использования везде ===
moon_radius = int(min(canvas_width, canvas_height) * 0.35)
moon_center_x = canvas_width // 2
moon_center_y = canvas_height // 2 - 250


# === ПОЛНАЯ ЛУНА (NURBS) ===

control_points_moon = np.array([
    [moon_center_x + moon_radius, moon_center_y],
    [moon_center_x + moon_radius, moon_center_y + moon_radius],
    [moon_center_x,               moon_center_y + moon_radius],
    [moon_center_x - moon_radius, moon_center_y + moon_radius],
    [moon_center_x - moon_radius, moon_center_y],
    [moon_center_x - moon_radius, moon_center_y - moon_radius],
    [moon_center_x,               moon_center_y - moon_radius],
    [moon_center_x + moon_radius, moon_center_y - moon_radius],
    [moon_center_x + moon_radius, moon_center_y]
], dtype=float)

moon_weights = np.array([1.0, np.sqrt(2)/2, 1.0, np.sqrt(2)/2,
                         1.0, np.sqrt(2)/2, 1.0, np.sqrt(2)/2, 1.0])
moon_knot = np.array([0, 0, 0,
                      0.25, 0.25,
                      0.5, 0.5,
                      0.75, 0.75,
                      1, 1, 1], dtype=float)
moon_degree = 2

moon_points = nurbs_curve(control_points_moon, moon_weights, moon_knot, moon_degree, num_samples=300)

if len(moon_points) > 0:
    fill_polygon_scanline(canvas, moon_points, moon_color)
    draw_bezier_with_line(canvas, moon_points, moon_color, linewidth=2)



# === РОВНО 3 ВПАДИНЫ-КРАТЕРА НА ВИДИМОЙ ЧАСТИ СЕРПА ===
crater_color = (180/255, 180/255, 180/255)

craters_drawn = 0
max_attempts = 200  # чтобы наверняка найти 3 подходящих точки
# === ВЫРЕЗАЕМ СЕРП: КРУГ ЦВЕТА ФОНА ПОВЕРХ ЛУНЫ ===
crescent_cut_radius = int(moon_radius * 0.95)
crescent_cut_center_x = moon_center_x + int(moon_radius * 0.35)
crescent_cut_center_y = moon_center_y
attempts = 0
while craters_drawn < 3 and attempts < max_attempts:
    attempts += 1

    # УГОЛ ТОЛЬКО ДЛЯ ЛЕВОЙ ПОЛОВИНЫ ЛУНЫ (от 90° до 270°)
    angle = random.uniform(np.pi/2, 3*np.pi/2)

    # Расстояние от центра – ближе к краю серпа
    crater_dist = random.uniform(moon_radius * 0.3, moon_radius * 0.8)
    crater_radius = random.uniform(moon_radius * 0.1, moon_radius * 0.2)

    x = moon_center_x + crater_dist * np.cos(angle)
    y = moon_center_y + crater_dist * np.sin(angle)

    # 1) Точка должна быть внутри основной луны
    if (x - moon_center_x)**2 + (y - moon_center_y)**2 > moon_radius**2:
        continue

    # 2) И ВНЕ вырезанного тёмного круга
    if (x - crescent_cut_center_x)**2 + (y - crescent_cut_center_y)**2 <= crescent_cut_radius**2:
        continue

    # 3) И явно на левой стороне (подстраховка)
    if x > moon_center_x:
        continue

    # Если все условия прошли — рисуем кратер
    fill_circle_scanline(canvas, x, y, crater_radius, crater_color)
    craters_drawn += 1


### для серпа

# fill_circle_scanline(
#     canvas,
#     crescent_cut_center_x,
#     crescent_cut_center_y,
#     crescent_cut_radius,
#     canvas_color
# )





# === ВЕТВИ ДЕРЕВЬЕВ (более прямые, от земли до середины луны) ===
branch_base_y = canvas_height - 50
mid_moon_y = moon_center_y 

num_branches = 5
branch_width = 8

for branch_idx in range(num_branches):
    start_x = canvas_width // 5 + (branch_idx * canvas_width // 5)
    start_x = max(100, min(canvas_width - 100, start_x))
    start_y = branch_base_y
    
    # Конечная точка близко к стартовой X (более прямая ветка)
    end_x = start_x + random.randint(-50, 50)
    end_y = mid_moon_y
    
    num_control_points = 3  # Меньше точек = более прямая
    control_points_branch = [[start_x, start_y]]
    
    # Одна промежуточная точка посередине
    mid_x = start_x + random.randint(-30, 30)
    mid_y = (start_y + end_y) // 2
    control_points_branch.append([mid_x, mid_y])
    
    control_points_branch.append([end_x, end_y])
    
    branch_points = bezier_curve(control_points_branch, num_samples=300)
    
    if len(branch_points) > 0:
        draw_bezier_with_line(canvas, branch_points, branch_color, branch_width)
    
    # Боковые ветвления
    for sub_branch in range(2):
        start_idx = random.randint(int(len(branch_points) * 0.3), int(len(branch_points) * 0.7))
        start_point = branch_points[start_idx]
        angle = random.choice([-1, 1])
        sub_length = random.randint(80, 150)
        sub_control_points = [
            start_point,
            [start_point[0] + angle * sub_length * 0.5, start_point[1] - sub_length * 0.3],
            [start_point[0] + angle * sub_length, start_point[1] - sub_length * 0.7]
        ]
        sub_branch_points = bezier_curve(sub_control_points, num_samples=100)
        if len(sub_branch_points) > 0:
            draw_bezier_with_line(canvas, sub_branch_points, branch_color, branch_width // 2)








# === ТЫКВА (NURBS) ===
pumpkin_center_x = canvas_width // 2
pumpkin_center_y = canvas_height // 2 + 400
pumpkin_radius = 180

pumpkin_cp = np.array([
    [pumpkin_center_x + pumpkin_radius * 1.1, pumpkin_center_y],
    [pumpkin_center_x + pumpkin_radius * 1.1, pumpkin_center_y + pumpkin_radius * 0.8],
    [pumpkin_center_x, pumpkin_center_y + pumpkin_radius * 1.2],
    [pumpkin_center_x - pumpkin_radius * 1.1, pumpkin_center_y + pumpkin_radius * 0.8],
    [pumpkin_center_x - pumpkin_radius * 1.1, pumpkin_center_y],
    [pumpkin_center_x - pumpkin_radius * 1.1, pumpkin_center_y - pumpkin_radius * 0.8],
    [pumpkin_center_x, pumpkin_center_y - pumpkin_radius * 1.2],
    [pumpkin_center_x + pumpkin_radius * 1.1, pumpkin_center_y - pumpkin_radius * 0.8],
    [pumpkin_center_x + pumpkin_radius * 1.1, pumpkin_center_y]
], dtype=float)

pumpkin_weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
pumpkin_knot = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=float)
pumpkin_degree = 2

pumpkin_points = nurbs_curve(pumpkin_cp, pumpkin_weights, pumpkin_knot, pumpkin_degree, num_samples=300)

if len(pumpkin_points) > 0:
    fill_polygon_scanline(canvas, pumpkin_points, pumpkin_color)
    draw_bezier_with_line(canvas, pumpkin_points, (200/255, 100/255, 0/255), linewidth=3)


# === ДОЛЬКИ ТЫКВЫ ===


def draw_pumpkin_ridge(canvas, p0, p1, p2, color, linewidth=8):
    """Рисует дольку-ребро кривой Безье второго порядка"""
    curve = bezier_curve([p0, p1, p2], num_samples=120)
    draw_bezier_with_line(canvas, curve, color, linewidth)

ridge_color = (200/255, 100/255, 0/255)  # тёмно-оранжевый

draw_pumpkin_ridge(
    canvas,
    p0 = [pumpkin_center_x - pumpkin_radius * 0.82,
          pumpkin_center_y - pumpkin_radius * 0.8],

    p1 = [pumpkin_center_x - pumpkin_radius * 1.05,   
          pumpkin_center_y - pumpkin_radius * 0.05],

    p2 = [pumpkin_center_x - pumpkin_radius * 0.78,
          pumpkin_center_y + pumpkin_radius * 0.8],

    color = ridge_color,
    linewidth = 14
)

draw_pumpkin_ridge(
    canvas,
    p0 = [pumpkin_center_x - pumpkin_radius * 0.45,
          pumpkin_center_y - pumpkin_radius * 0.97],

    p1 = [pumpkin_center_x - pumpkin_radius * 0.80,   # ← ДАЛЬШЕ ВЛЕВО
          pumpkin_center_y - pumpkin_radius * 0.10],

    p2 = [pumpkin_center_x - pumpkin_radius * 0.40,
          pumpkin_center_y + pumpkin_radius * 0.8],

    color = ridge_color,
    linewidth = 10
)
draw_pumpkin_ridge(
    canvas,
    p0 = [pumpkin_center_x + pumpkin_radius * 0.45,
          pumpkin_center_y - pumpkin_radius * 0.97],

    p1 = [pumpkin_center_x + pumpkin_radius * 0.8,   # → ДАЛЬШЕ ВПРАВО
          pumpkin_center_y - pumpkin_radius * 0.10],

    p2 = [pumpkin_center_x + pumpkin_radius * 0.40,
          pumpkin_center_y + pumpkin_radius * 0.8],

    color = ridge_color,
    linewidth = 10
)

draw_pumpkin_ridge(
    canvas,
    p0 = [pumpkin_center_x + pumpkin_radius * 0.82,
          pumpkin_center_y - pumpkin_radius * 0.8],

    p1 = [pumpkin_center_x + pumpkin_radius * 1.05,   # → ВНЕШНЕЕ ОТКЛОНЕНИЕ
          pumpkin_center_y - pumpkin_radius * 0.05],

    p2 = [pumpkin_center_x + pumpkin_radius * 0.78,
          pumpkin_center_y + pumpkin_radius * 0.8],

    color = ridge_color,
    linewidth = 14
)

draw_pumpkin_ridge(
    canvas,
    p0 = [pumpkin_center_x + pumpkin_radius * 0.82,
          pumpkin_center_y - pumpkin_radius * 0.8],

    p1 = [pumpkin_center_x + pumpkin_radius * 1.05,   # → ВНЕШНЕЕ ОТКЛОНЕНИЕ
          pumpkin_center_y - pumpkin_radius * 0.05],

    p2 = [pumpkin_center_x + pumpkin_radius * 0.78,
          pumpkin_center_y + pumpkin_radius * 0.8],

    color = ridge_color,
    linewidth = 14
)


draw_pumpkin_ridge(
    canvas,
    p0=[pumpkin_center_x, pumpkin_center_y - pumpkin_radius ],
    p1=[pumpkin_center_x, pumpkin_center_y - pumpkin_radius * 0.18],
    p2=[pumpkin_center_x, pumpkin_center_y + pumpkin_radius ],
    color=ridge_color,
    linewidth=14   # центральное толще всех
)


# === ГОРИЗОНТАЛЬНАЯ КОЧЕРЫЖКА ===
stem_cp = np.array([
    [pumpkin_center_x, pumpkin_center_y - pumpkin_radius * 1.1],
    [pumpkin_center_x + 30, pumpkin_center_y - pumpkin_radius * 1.15],
    [pumpkin_center_x + 60, pumpkin_center_y - pumpkin_radius * 1.1],
    [pumpkin_center_x + 90, pumpkin_center_y - pumpkin_radius * 1.0],
    [pumpkin_center_x + 110, pumpkin_center_y - pumpkin_radius * 0.9]
], dtype=float)

stem_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
stem_knot = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)
stem_degree = 3

stem_points = nurbs_curve(stem_cp, stem_weights, stem_knot, stem_degree, num_samples=150)
if len(stem_points) > 0:
    draw_bezier_with_line(canvas, stem_points, stem_color, linewidth=16)

# Конец кочерыжки (закруглённый)
stem_top_x = stem_points[-1][0]
stem_top_y = stem_points[-1][1]
stem_top_radius = 20

fill_circle_scanline(canvas, stem_top_x, stem_top_y, stem_top_radius, stem_color)

# === ГЛАЗА ===
eye_left_x = pumpkin_center_x - 60
eye_left_y = pumpkin_center_y - 40
eye_right_x = pumpkin_center_x + 60
eye_right_y = pumpkin_center_y - 40

left_eye_points = np.array([
    [eye_left_x - 25, eye_left_y + 20],
    [eye_left_x + 25, eye_left_y + 20],
    [eye_left_x, eye_left_y - 25]
])
fill_polygon_scanline(canvas, left_eye_points, (0, 0, 0))

right_eye_points = np.array([
    [eye_right_x - 25, eye_right_y + 20],
    [eye_right_x + 25, eye_right_y + 20],
    [eye_right_x, eye_right_y - 25]
])
fill_polygon_scanline(canvas, right_eye_points, (0, 0, 0))



# === НОС ===
nose_x = pumpkin_center_x
nose_y = pumpkin_center_y
nose_size = 30

nose_points = np.array([
    [nose_x - nose_size, nose_y - nose_size],
    [nose_x + nose_size, nose_y - nose_size],
    [nose_x, nose_y + nose_size]
])
fill_polygon_scanline(canvas, nose_points, (0, 0, 0))

# === ОТКРЫТЫЙ РОТ С ЗУБАМИ ===

# верхняя и нижняя границы рта (кривые Безье 2-го порядка)
mouth_top_cp = [
    [pumpkin_center_x - 90, pumpkin_center_y + 40],
    [pumpkin_center_x,      pumpkin_center_y + 10],
    [pumpkin_center_x + 90, pumpkin_center_y + 40]
]

mouth_bottom_cp = [
    [pumpkin_center_x - 90, pumpkin_center_y + 40],
    [pumpkin_center_x,      pumpkin_center_y + 120],
    [pumpkin_center_x + 90, pumpkin_center_y + 40]
]

mouth_top_curve = bezier_curve(mouth_top_cp, num_samples=140)
mouth_bottom_curve = bezier_curve(mouth_bottom_cp, num_samples=140)

# замкнутый многоугольник для полости рта
mouth_polygon = np.vstack([mouth_top_curve, mouth_bottom_curve[::-1]])
mouth_inside_color = (10/255, 10/255, 10/255)
fill_polygon_scanline(canvas, mouth_polygon, mouth_inside_color)

# --- зубы ---
tooth_color = (245/255, 245/255, 220/255)
upper_y = pumpkin_center_y + 40
lower_y = pumpkin_center_y + 90
tooth_w = 26
tooth_h = 28

# верхние зубы (смотрят вниз)
for cx in [pumpkin_center_x - 60, pumpkin_center_x - 20,
           pumpkin_center_x + 20, pumpkin_center_x + 60]:
    tooth_pts = np.array([
        [cx - tooth_w/2, upper_y],
        [cx + tooth_w/2, upper_y],
        [cx,             upper_y + tooth_h]
    ])
    fill_polygon_scanline(canvas, tooth_pts, tooth_color)

# нижние зубы (смотрят вверх)
for cx in [pumpkin_center_x - 40, pumpkin_center_x,
           pumpkin_center_x + 40]:
    tooth_pts = np.array([
        [cx - tooth_w/2, lower_y],
        [cx + tooth_w/2, lower_y],
        [cx,             lower_y - tooth_h]
    ])
    fill_polygon_scanline(canvas, tooth_pts, tooth_color)


# # === РОТ ===
# mouth_cp = np.array([
#     [pumpkin_center_x - 80, pumpkin_center_y + 60],
#     [pumpkin_center_x, pumpkin_center_y + 100],
#     [pumpkin_center_x + 80, pumpkin_center_y + 60]
# ], dtype=float)

# mouth_weights = np.array([1.0, 1.0, 1.0])
# mouth_knot = np.array([0, 0, 0, 1, 1, 1], dtype=float)
# mouth_degree = 2

# mouth_points = nurbs_curve(mouth_cp, mouth_weights, mouth_knot, mouth_degree, num_samples=150)
# if len(mouth_points) > 0:
#     draw_bezier_with_line(canvas, mouth_points, (0, 0, 0), linewidth=8)



# === ВОРОН НА КОЧЕРЫЖКЕ ===
raven_x = stem_top_x - 50
raven_y = stem_top_y - stem_top_radius - 200

# Параметры ворона
raven_body_width = 200
raven_body_height = 160
raven_head_radius = 70

# 1. ТЕЛО ВОРОНА
body_control_points = [
    [raven_x, raven_y - raven_body_height * 0.5],
    [raven_x + raven_body_width * 0.6, raven_y - raven_body_height * 0.4],
    [raven_x + raven_body_width * 0.8, raven_y],
    [raven_x + raven_body_width * 0.6, raven_y + raven_body_height * 0.4],
    [raven_x, raven_y + raven_body_height * 0.5],
    [raven_x - raven_body_width * 0.6, raven_y + raven_body_height * 0.4],
    [raven_x - raven_body_width * 0.8, raven_y],
    [raven_x - raven_body_width * 0.6, raven_y - raven_body_height * 0.4],
    [raven_x, raven_y - raven_body_height * 0.5]
]

body_points = bezier_curve(body_control_points, num_samples=300)
fill_polygon_scanline(canvas, body_points, raven_color)

# 2. ГОЛОВА ВОРОНА
head_center_x = raven_x + raven_body_width * 0.5
head_center_y = raven_y - raven_body_height * 0.4

fill_circle_scanline(canvas, head_center_x, head_center_y, raven_head_radius, raven_color)

# 3. КРЫЛО ВОРОНА (левое)
wing_control_points = [
    [raven_x - raven_body_width * 0.3, raven_y - raven_body_height * 0.2],
    [raven_x - raven_body_width * 0.9, raven_y - raven_body_height * 0.6],
    [raven_x - raven_body_width * 1.3, raven_y - raven_body_height * 0.3],
    [raven_x - raven_body_width * 1.4, raven_y + raven_body_height * 0.25],
    [raven_x - raven_body_width * 0.7, raven_y + raven_body_height * 0.4]
]

wing_points = bezier_curve(wing_control_points, num_samples=250)
wing_color = (60/255, 60/255, 60/255)
fill_polygon_scanline(canvas, wing_points, wing_color)
draw_bezier_with_line(canvas, wing_points, (150/255, 150/255, 150/255), linewidth=3)


# 4. ГЛА3 ВОРОНА
eye_x = head_center_x + raven_head_radius * 0.4
eye_y = head_center_y - raven_head_radius * 0.2
eye_radius = 20

fill_circle_scanline(canvas, eye_x, eye_y, eye_radius, eye_color)

# Зрачок
pupil_radius = 8
fill_circle_scanline(canvas, eye_x + eye_radius * 0.5, eye_y, pupil_radius, (0, 0, 0))

# 5. КЛЮВ ВОРОНА (на краю головы)
beak_control_points = [
    [head_center_x + raven_head_radius * 0.95, head_center_y - raven_head_radius * 0.1],
    [head_center_x + raven_head_radius * 1.4, head_center_y - raven_head_radius * 0.05],
    [head_center_x + raven_head_radius * 1.6, head_center_y]
]

beak_points = bezier_curve(beak_control_points, num_samples=120)
draw_bezier_with_line(canvas, beak_points, beak_color, linewidth=8)

beak_bottom_control_points = [
    [head_center_x + raven_head_radius * 0.95, head_center_y + raven_head_radius * 0.1],
    [head_center_x + raven_head_radius * 1.4, head_center_y + raven_head_radius * 0.05],
    [head_center_x + raven_head_radius * 1.6, head_center_y + raven_head_radius * 0.1]
]

beak_bottom_points = bezier_curve(beak_bottom_control_points, num_samples=120)
draw_bezier_with_line(canvas, beak_bottom_points, beak_color, linewidth=8)

# 6. ЛАПКИ ВОРОНА (две серые)
# Левая лапка
left_leg_control_points = [
    [raven_x - raven_body_width * 0.3, raven_y + raven_body_height * 0.2],
    [raven_x - raven_body_width * 0.4, raven_y + raven_body_height * 0.7],
    [raven_x - raven_body_width * 0.25, raven_y + raven_body_height * 0.85]
]

left_leg_points = bezier_curve(left_leg_control_points, num_samples=150)
draw_bezier_with_line(canvas, left_leg_points, beak_color, linewidth=7)

# Пальцы левой лапки
for toe_angle in [-0.3, 0, 0.3]:
    toe_control_points = [
        left_leg_points[-1],
        [left_leg_points[-1][0] + toe_angle * 35, left_leg_points[-1][1] + 25],
        [left_leg_points[-1][0] + toe_angle * 50, left_leg_points[-1][1] + 40]
    ]
    toe_points = bezier_curve(toe_control_points, num_samples=80)
    draw_bezier_with_line(canvas, toe_points, beak_color, linewidth=5)

# Правая лапка
right_leg_control_points = [
    [raven_x + raven_body_width * 0.1, raven_y + raven_body_height * 0.3],
    [raven_x + raven_body_width * 0.2, raven_y + raven_body_height * 0.7],
    [raven_x + raven_body_width * 0.3, raven_y + raven_body_height * 0.95]
]

right_leg_points = bezier_curve(right_leg_control_points, num_samples=150)
draw_bezier_with_line(canvas, right_leg_points, beak_color, linewidth=7)

# Пальцы правой лапки
for toe_angle in [-0.3, 0, 0.3]:
    toe_control_points = [
        right_leg_points[-1],
        [right_leg_points[-1][0] + toe_angle * 35, right_leg_points[-1][1] + 25],
        [right_leg_points[-1][0] + toe_angle * 50, right_leg_points[-1][1] + 40]
    ]
    toe_points = bezier_curve(toe_control_points, num_samples=80)
    draw_bezier_with_line(canvas, toe_points, beak_color, linewidth=5)

# === ПЕРЬЯ НА КРЫЛЕ ВОРОНА (Bezier 2-го порядка) ===
feather_color = (40/255, 40/255, 40/255)

feather_cps = [
    # перо ближе к хвосту
    [
        [raven_x - raven_body_width * 0.95, raven_y + raven_body_height * 0.00],
        [raven_x - raven_body_width * 1.15, raven_y + raven_body_height * 0.20],
        [raven_x - raven_body_width * 0.80, raven_y + raven_body_height * 0.30],
    ],
    # среднее перо
    [
        [raven_x - raven_body_width * 0.75, raven_y + raven_body_height * 0.10],
        [raven_x - raven_body_width * 0.95, raven_y + raven_body_height * 0.32],
        [raven_x - raven_body_width * 0.55, raven_y + raven_body_height * 0.38],
    ],
    # перо ближе к телу
    [
        [raven_x - raven_body_width * 0.55, raven_y + raven_body_height * 0.15],
        [raven_x - raven_body_width * 0.70, raven_y + raven_body_height * 0.35],
        [raven_x - raven_body_width * 0.35, raven_y + raven_body_height * 0.40],
    ],
]

for cps in feather_cps:
    f_curve = bezier_curve(cps, num_samples=100)  # порядок 2: 3 контрольные точки
    draw_bezier_with_line(canvas, f_curve, feather_color, linewidth=5)


# Показываем результат
fig, ax = plt.subplots(figsize=(canvas_width/100, canvas_height/100), dpi=100)
ax.imshow(canvas)
ax.axis('off')
plt.tight_layout(pad=0)
plt.savefig('halloween_pumpkin_moon.png', dpi=100, bbox_inches='tight')
plt.show()

print("Хэллоуин рисунок готов!")
print(f"Размер холста: {canvas.shape}")
