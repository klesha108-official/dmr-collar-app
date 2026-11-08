"""線画の読み込みと前処理（ローカル処理・無料）。"""
import cv2
import numpy as np


def decode_image(data: bytes, max_side=2000):
    """バイト列から画像(BGR)を読み込み、長辺を max_side に縮小する。"""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("画像を読み込めませんでした")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def extract_lines(bgr, sensitivity=0.90, gain=2.5):
    """紙の灰色・影を除去し、線の濃さ(0=紙, 1=線)を返す。

    背景の明るさをメディアンぼかしで推定して割り算する（照明ムラ補正）。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(31, (min(gray.shape) // 20) | 1)
    bg = cv2.medianBlur(gray, k).astype(np.float32)
    ratio = np.clip(gray.astype(np.float32) / np.maximum(bg, 1), 0, 1.2)
    darkness = np.clip((sensitivity - ratio) / sensitivity, 0, 1)
    return np.clip(darkness * gain, 0, 1)


def clean_lineart(darkness, line_gray=40):
    """AI に渡す用の線画画像(RGB)。紙は真っ白、線は濃いグレー。"""
    v = 255 - darkness * (255 - line_gray)
    v = np.clip(v, 0, 255).astype(np.uint8)
    return np.dstack([v, v, v])
