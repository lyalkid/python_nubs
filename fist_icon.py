"""
Рисует простую иконку сжатого кулака (контур, как на примере).
"""

import os
from PIL import Image, ImageDraw


def rounded(draw: ImageDraw.ImageDraw, xy, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_fist(size=512, save_path="outputs/fist_icon.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    outline = "black"
    lw = max(4, size // 64)

    # Основная ладонь
    rounded(d, (150, 240, 380, 450), radius=40, fill=None, outline=outline, width=lw)

    # Четыре пальца
    finger_w = 55
    finger_h = 130
    start_y = 140
    xs = [165, 225, 285, 345]
    for x in xs:
        rounded(d, (x, start_y, x + finger_w, start_y + finger_h), radius=25, fill=None, outline=outline, width=lw)

    # Вырез для суставов (легкий изгиб вниз)
    d.line([(165, 240), (400, 260)], fill=outline, width=lw)

    # Большой палец
    rounded(d, (120, 250, 210, 370), radius=30, fill=None, outline=outline, width=lw)
    d.line([(140, 320), (220, 310)], fill=outline, width=lw)

    img.save(save_path)
    return img


if __name__ == "__main__":
    img = draw_fist()
    print("Сохранено:", "outputs/fist_icon.png")
