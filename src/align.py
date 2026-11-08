"""AI が着色した画像に、元の線画を位置合わせする。"""
import cv2
import numpy as np

from lineart import extract_lines


def _ai_line_map(ai_bgr):
    """AI 着色画の「線らしさ」。色のついた面を線と誤認しないよう彩度で割り引く。"""
    hsv = cv2.cvtColor(ai_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(ai_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 51).astype(np.float32)
    d = (1 - gray.astype(np.float32) / np.maximum(bg, 1)) * 2.5
    return np.clip(d * (1 - hsv[..., 1] / 255.0 * 0.7), 0, 1)


def align_lines(orig_bgr, ai_bgr, sensitivity=0.90):
    """元の線画の濃さを、AI 画像と同じ大きさ・位置に合わせて返す。

    戻り値: (aligned_darkness, 一致度 0〜1, 縦横比の差[割合])
    粗い解像度から順に ECC（相関最大化）で射影変換を求める。失敗時は拡縮のみ。
    """
    h, w = ai_bgr.shape[:2]
    d_orig = extract_lines(orig_bgr, sensitivity)
    d_o = cv2.resize(d_orig, (w, h), interpolation=cv2.INTER_AREA)
    aspect_diff = abs((orig_bgr.shape[1] / orig_bgr.shape[0]) / (w / h) - 1)

    a = cv2.GaussianBlur(_ai_line_map(ai_bgr), (0, 0), 3).astype(np.float32)
    b = cv2.GaussianBlur(d_o, (0, 0), 3).astype(np.float32)

    warp = np.eye(3, dtype=np.float32)
    cc = 0.0
    try:
        for sc in (0.25, 0.5, 1.0):
            sa = cv2.resize(a, None, fx=sc, fy=sc)
            sb = cv2.resize(b, None, fx=sc, fy=sc)
            wm = warp.copy()
            wm[:2, 2] *= sc
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            cc, wm = cv2.findTransformECC(sa, sb, wm, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
            wm[:2, 2] /= sc
            warp = wm
    except cv2.error:
        return d_o, 0.0, aspect_diff

    aligned = cv2.warpPerspective(d_o, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned, float(cc), aspect_diff
