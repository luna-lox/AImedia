#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подбор HSV-порогов для tracker.py.

Запуск:
    python3 calibrate.py input.mp4

Управление:
    ползунки H/S/V   — границы cv2.inRange()
    клик по кадру    — взять цвет пикселя и построить диапазон вокруг него
    пробел           — пауза / продолжить
    a / d            — шаг на кадр назад / вперёд (в паузе)
    s                — напечатать готовую команду для tracker.py
    q или Esc        — выход (команда печатается автоматически)
"""

import argparse
import sys

import cv2
import numpy as np

WINDOW = "calibrate: H/S/V + click (q - exit)"
BARS = [("H min", 179), ("H max", 179), ("S min", 255), ("S max", 255), ("V min", 255), ("V max", 255)]


def nothing(_):
    pass


def read_bars():
    return [cv2.getTrackbarPos(name, WINDOW) for name, _ in BARS]


def set_bars(values):
    for (name, _), value in zip(BARS, values):
        cv2.setTrackbarPos(name, WINDOW, int(value))


def on_mouse(event, x, y, flags, param):
    """Клик по кадру: берём медианный HSV в окне 9x9 и расширяем до диапазона."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    frame = param.get("frame")
    if frame is None:
        return
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - 4), min(w, x + 5)
    y0, y1 = max(0, y - 4), min(h, y + 5)
    patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat, val = np.median(patch, axis=0)
    set_bars([max(0, hue - 10), min(179, hue + 10),
              max(0, sat - 70), 255,
              max(0, val - 70), 255])
    print("пиксель (%d, %d): H=%d S=%d V=%d" % (x, y, hue, sat, val))


def command_for(path, values):
    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = values
    return ("python3 tracker.py %s -o result.mp4 "
            "--hsv-lower %d,%d,%d --hsv-upper %d,%d,%d" % (path, h_lo, s_lo, v_lo, h_hi, s_hi, v_hi))


def main():
    parser = argparse.ArgumentParser(description="Интерактивный подбор HSV-порогов.")
    parser.add_argument("input", help="видео (или индекс камеры, например 0)")
    parser.add_argument("--scale", type=float, default=0.6, help="масштаб окна предпросмотра")
    args = parser.parse_args()

    source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit("Не удалось открыть источник видео: %s" % args.input)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for name, maximum in BARS:
        cv2.createTrackbar(name, WINDOW, 0, maximum, nothing)
    set_bars([5, 20, 120, 255, 110, 255])          # стартовый диапазон — оранжевый

    state = {"frame": None}
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    paused = False
    frame = None
    while True:
        if not paused or frame is None:
            ok, new_frame = cap.read()
            if not ok:                               # видео кончилось — перематываем
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = new_frame
            if args.scale != 1.0:
                frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            state["frame"] = frame

        values = read_bars()
        h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = values
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        if h_lo <= h_hi:
            mask = cv2.inRange(hsv, np.array((h_lo, s_lo, v_lo), np.uint8),
                                    np.array((h_hi, s_hi, v_hi), np.uint8))
        else:
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array((h_lo, s_lo, v_lo), np.uint8), np.array((179, s_hi, v_hi), np.uint8)),
                cv2.inRange(hsv, np.array((0, s_lo, v_lo), np.uint8), np.array((h_hi, s_hi, v_hi), np.uint8)))

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        preview = np.hstack([frame, cv2.bitwise_and(frame, frame, mask=mask),
                             cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.putText(preview, "H %d-%d  S %d-%d  V %d-%d" % (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi),
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            print(command_for(args.input, values))
        elif key == ord("d") and paused:
            paused = False
            ok, frame = cap.read()
            paused = True
            if ok:
                if args.scale != 1.0:
                    frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
                state["frame"] = frame
        elif key == ord("a") and paused:
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
            ok, frame = cap.read()
            if ok:
                if args.scale != 1.0:
                    frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
                state["frame"] = frame

    cap.release()
    cv2.destroyAllWindows()
    print("\nИтоговые пороги — запускайте трекер так:\n" + command_for(args.input, values))
    return 0


if __name__ == "__main__":
    sys.exit(main())
