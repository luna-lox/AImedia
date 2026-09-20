#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Трекинг объекта классическими методами компьютерного зрения (OpenCV).

Пайплайн для каждого кадра:
  1) размытие (GaussianBlur) и перевод в HSV либо в оттенки серого;
  2) бинаризация:  cv2.inRange()  (режим color) или cv2.threshold() (bright/dark);
  3) морфология MORPH_OPEN + MORPH_CLOSE — убираем шум и залепляем дырки;
  4) cv2.findContours() -> самый большой контур,
     cv2.moments() -> центроид объекта (cx, cy);
  5) центроид кладётся в «след», который рисуется примитивами
     (drawMarker / line / circle / rectangle) поверх текущего кадра.

Примеры запуска:
    python3 tracker.py input.mp4 -o result.mp4 --color orange
    python3 tracker.py input.mp4 -o result.mp4 --hsv-lower 15,120,120 --hsv-upper 35,255,255
    python3 tracker.py input.mp4 -o result.mp4 --mode bright --thresh 220 --debug
"""

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np

# Готовые диапазоны HSV (OpenCV: H 0..179, S 0..255, V 0..255).
# У красного цвета тон «заворачивается» через 0, поэтому lower_H > upper_H —
# этот случай обрабатывается в build_mask() двумя вызовами inRange().
PRESETS = {
    "red":    ((170, 110, 70),  (10, 255, 255)),
    "orange": ((5, 120, 110),   (20, 255, 255)),
    "yellow": ((22, 100, 110),  (35, 255, 255)),
    "green":  ((40, 80, 60),    (85, 255, 255)),
    "cyan":   ((85, 90, 80),    (100, 255, 255)),
    "blue":   ((100, 120, 60),  (130, 255, 255)),
    "purple": ((130, 70, 60),   (160, 255, 255)),
    "pink":   ((160, 80, 120),  (175, 255, 255)),
}

MARKERS = {
    "cross":  cv2.MARKER_CROSS,
    "tilted": cv2.MARKER_TILTED_CROSS,
    "star":   cv2.MARKER_STAR,
    "square": cv2.MARKER_SQUARE,
    "diamond": cv2.MARKER_DIAMOND,
    "triangle": cv2.MARKER_TRIANGLE_UP,
}

TRAIL_OLD = (255, 180, 60)    # BGR: голубой — старые отметки
TRAIL_NEW = (60, 60, 255)     # BGR: красный — свежие отметки
CURRENT   = (0, 255, 255)     # BGR: жёлтый — текущее положение


def parse_triplet(text, name):
    """'15,120,120' -> (15, 120, 120)"""
    parts = text.replace(" ", "").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("%s: ожидается 'H,S,V', получено %r" % (name, text))
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("%s: значения должны быть целыми, получено %r" % (name, text))


def odd(value):
    """Ядро фильтра должно быть нечётным и >= 1."""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def build_mask(frame, args):
    """Кадр -> бинарная маска объекта (inRange или threshold + морфология)."""
    if args.blur > 1:
        frame = cv2.GaussianBlur(frame, (odd(args.blur), odd(args.blur)), 0)

    if args.mode == "color":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo, hi = args.hsv_lower, args.hsv_upper
        if lo[0] <= hi[0]:
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        else:
            # тон «заворачивается» через 0 (красный): объединяем два диапазона
            lower_part = cv2.inRange(hsv, np.array((lo[0], lo[1], lo[2]), np.uint8),
                                          np.array((179, hi[1], hi[2]), np.uint8))
            upper_part = cv2.inRange(hsv, np.array((0, lo[1], lo[2]), np.uint8),
                                          np.array((hi[0], hi[1], hi[2]), np.uint8))
            mask = cv2.bitwise_or(lower_part, upper_part)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flag = cv2.THRESH_BINARY if args.mode == "bright" else cv2.THRESH_BINARY_INV
        _, mask = cv2.threshold(gray, args.thresh, 255, flag)

    if args.morph > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(args.morph), odd(args.morph)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def find_object(mask, min_area):
    """Самый большой контур маски -> (центроид, bbox, площадь) или None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None

    cx = int(round(m["m10"] / m["m00"]))
    cy = int(round(m["m01"] / m["m00"]))
    return (cx, cy), cv2.boundingRect(contour), area


def mix(color_a, color_b, t):
    """Линейная интерполяция двух BGR-цветов, t in [0, 1]."""
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(color_a, color_b))


def draw_trail(frame, trail, args):
    """Рисует путь объекта: линия между кадрами + отметки-крестики."""
    points = list(trail)
    total = max(len(points) - 1, 1)

    # 1. соединяем последовательные позиции линией (разрывы там, где объект потерян)
    for i in range(1, len(points)):
        prev, curr = points[i - 1], points[i]
        if prev is None or curr is None:
            continue
        cv2.line(frame, prev, curr, mix(TRAIL_OLD, TRAIL_NEW, i / total), 2, cv2.LINE_AA)

    # 2. отметки на позициях объекта на предыдущих кадрах
    last = len(points) - 1
    for i, point in enumerate(points):
        if point is None or (last - i) % args.mark_every != 0:
            continue
        color = mix(TRAIL_OLD, TRAIL_NEW, i / total)
        cv2.drawMarker(frame, point, color, MARKERS[args.marker],
                       markerSize=args.marker_size, thickness=2, line_type=cv2.LINE_AA)


def draw_current(frame, center, bbox):
    """Подсветка текущего положения объекта."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), CURRENT, 2)
    cv2.circle(frame, center, 6, CURRENT, -1, cv2.LINE_AA)
    cv2.circle(frame, center, 14, CURRENT, 2, cv2.LINE_AA)


def draw_hud(frame, lines):
    """Текстовая панель в левом верхнем углу (с тенью для читаемости)."""
    for i, text in enumerate(lines):
        org = (12, 28 + i * 26)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def side_by_side(frame, mask):
    """Кадр + бинарная маска рядом — удобно для отладки порогов."""
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_bgr, "MASK", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return np.hstack([frame, mask_bgr])


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Трекинг объекта по цвету/яркости с отрисовкой пути (OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="исходное видео (или индекс камеры, например 0)")
    p.add_argument("-o", "--output", default=None, help="куда сохранить результат (.mp4)")

    p.add_argument("--mode", choices=["color", "bright", "dark"], default="color",
                   help="color = inRange по HSV, bright/dark = threshold по яркости")
    p.add_argument("--color", choices=sorted(PRESETS), default="orange",
                   help="готовый HSV-пресет для режима color")
    p.add_argument("--hsv-lower", type=lambda s: parse_triplet(s, "--hsv-lower"), default=None,
                   help="нижняя граница HSV 'H,S,V' (перекрывает --color)")
    p.add_argument("--hsv-upper", type=lambda s: parse_triplet(s, "--hsv-upper"), default=None,
                   help="верхняя граница HSV 'H,S,V'")
    p.add_argument("--thresh", type=int, default=200, help="порог яркости для режимов bright/dark")

    p.add_argument("--min-area", type=int, default=150, help="минимальная площадь контура, px")
    p.add_argument("--blur", type=int, default=5, help="ядро GaussianBlur (0 — выключить)")
    p.add_argument("--morph", type=int, default=5, help="ядро морфологии (0 — выключить)")
    p.add_argument("--smooth", type=float, default=0.35,
                   help="сглаживание центроида, 0 — выключено, ближе к 1 — сильнее")

    p.add_argument("--trail", type=int, default=0, help="длина следа в кадрах (0 — весь путь)")
    p.add_argument("--mark-every", type=int, default=3, help="ставить отметку каждый N-й кадр")
    p.add_argument("--marker", choices=sorted(MARKERS), default="tilted", help="форма отметки")
    p.add_argument("--marker-size", type=int, default=12, help="размер отметки, px")

    p.add_argument("--scale", type=float, default=1.0, help="масштабирование кадра")
    p.add_argument("--start", type=float, default=0.0, help="начать с этой секунды")
    p.add_argument("--end", type=float, default=None, help="закончить на этой секунде")
    p.add_argument("--debug", action="store_true", help="дописать справа бинарную маску")
    p.add_argument("--show", action="store_true", help="показывать окно во время обработки")
    p.add_argument("--fourcc", default="mp4v", help="кодек VideoWriter")

    args = p.parse_args(argv)

    preset_lo, preset_hi = PRESETS[args.color]
    if args.hsv_lower is None:
        args.hsv_lower = preset_lo
    if args.hsv_upper is None:
        args.hsv_upper = preset_hi
    args.mark_every = max(1, args.mark_every)
    args.smooth = min(max(args.smooth, 0.0), 0.95)
    return args


def main(argv=None):
    args = parse_args(argv)

    source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit("Не удалось открыть источник видео: %s" % args.input)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 25.0
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    trail = deque(maxlen=args.trail if args.trail > 0 else None)
    writer = None
    smoothed = None
    frame_idx = 0
    detected = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.end is not None and cap.get(cv2.CAP_PROP_POS_MSEC) > args.end * 1000.0:
            break
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

        frame_idx += 1
        mask = build_mask(frame, args)
        found = find_object(mask, args.min_area)

        if found is None:
            smoothed = None
            trail.append(None)                       # разрыв следа
            hud = ["frame %d" % frame_idx, "OBJECT LOST"]
        else:
            detected += 1
            center, bbox, area = found
            if smoothed is None or args.smooth == 0:
                smoothed = (float(center[0]), float(center[1]))
            else:
                a = 1.0 - args.smooth
                smoothed = (smoothed[0] + a * (center[0] - smoothed[0]),
                            smoothed[1] + a * (center[1] - smoothed[1]))
            point = (int(round(smoothed[0])), int(round(smoothed[1])))
            trail.append(point)
            hud = ["frame %d" % frame_idx,
                   "x=%d  y=%d" % point,
                   "area=%d px" % int(area)]

        draw_trail(frame, trail, args)
        if found is not None:
            draw_current(frame, trail[-1], found[1])
        draw_hud(frame, hud)

        out_frame = side_by_side(frame, mask) if args.debug else frame

        if args.output:
            if writer is None:
                h, w = out_frame.shape[:2]
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*args.fourcc), fps, (w, h))
                if not writer.isOpened():
                    sys.exit("Не удалось создать файл %s (попробуйте другой --fourcc)" % args.output)
            writer.write(out_frame)

        if args.show:
            cv2.imshow("tracker (q - выход)", out_frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        if frame_idx % 50 == 0:
            print("обработано кадров: %d" % frame_idx, flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    rate = 100.0 * detected / frame_idx if frame_idx else 0.0
    print("Готово: %d кадров, объект найден на %d (%.1f%%)" % (frame_idx, detected, rate))
    if args.output:
        size_mb = os.path.getsize(args.output) / 1e6 if os.path.exists(args.output) else 0.0
        print("Результат: %s (%.1f МБ)" % (args.output, size_mb))
    if frame_idx and rate < 50:
        print("Подсказка: объект найден меньше чем на половине кадров — "
              "подберите пороги через calibrate.py или уменьшите --min-area")
    return 0


if __name__ == "__main__":
    sys.exit(main())
