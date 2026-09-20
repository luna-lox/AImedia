#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор синтетического ролика для проверки трекера.

Оранжевый мяч летает по экрану на пёстром фоне, есть посторонние цветные
объекты-отвлекатели. Нужен только для самопроверки пайплайна — для сдачи
работы снимите или скачайте своё видео.

    python3 make_sample_video.py sample.mp4 --seconds 12
"""

import argparse
import math

import cv2
import numpy as np


def background(width, height, rng):
    """Серый фон с шумом и статичными цветными пятнами-отвлекателями."""
    bg = np.full((height, width, 3), 60, np.uint8)
    bg = cv2.add(bg, rng.integers(0, 35, (height, width, 3), dtype=np.uint8))
    for _ in range(14):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        color = tuple(int(c) for c in rng.integers(40, 160, 3))
        cv2.rectangle(bg, (x, y), (x + int(rng.integers(30, 120)), y + int(rng.integers(30, 120))), color, -1)
    cv2.circle(bg, (int(width * 0.18), int(height * 0.75)), 40, (40, 200, 40), -1)     # зелёный отвлекатель
    cv2.circle(bg, (int(width * 0.82), int(height * 0.22)), 34, (200, 60, 40), -1)     # синий отвлекатель
    return cv2.GaussianBlur(bg, (5, 5), 0)


def main():
    parser = argparse.ArgumentParser(description="Синтетическое видео с движущимся оранжевым мячом.")
    parser.add_argument("output", nargs="?", default="sample.mp4")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", default="960x540")
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    rng = np.random.default_rng(7)
    bg = background(width, height, rng)

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    total = int(args.seconds * args.fps)
    for i in range(total):
        t = i / args.fps
        frame = bg.copy()
        # траектория-«восьмёрка» с дрейфом
        x = width * (0.5 + 0.34 * math.sin(1.1 * t))
        y = height * (0.5 + 0.30 * math.sin(2.2 * t + 0.6) * math.cos(0.4 * t))
        radius = int(22 + 4 * math.sin(3.0 * t))
        cv2.circle(frame, (int(x), int(y)), radius, (30, 140, 245), -1, cv2.LINE_AA)   # оранжевый мяч
        cv2.circle(frame, (int(x) - 6, int(y) - 6), max(4, radius // 3), (120, 200, 255), -1, cv2.LINE_AA)
        frame = cv2.add(frame, rng.integers(0, 12, (height, width, 3), dtype=np.uint8))
        writer.write(frame)

    writer.release()
    print("Записано %s: %d кадров, %dx%d @ %d fps" % (args.output, total, width, height, args.fps))


if __name__ == "__main__":
    main()
