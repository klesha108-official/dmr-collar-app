"""AI 着色画の仕上げ（線の重ね合わせ・色調整・背景）。毎回同じ条件でかけて品質をそろえる。"""
import cv2
import numpy as np
from scipy import ndimage as ndi

PAPERS = {"そのまま": None, "紙白": (250, 249, 246), "クリーム": (246, 240, 226),
          "薄いグラデーション": "gradient"}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _background_mask(img):
    """画像の外周に接する「背景（紙）らしい」領域のマスク(0〜1)。"""
    h, w = img.shape[:2]
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32)
    ref = np.median(border, axis=0)
    dist = np.linalg.norm(cv2.GaussianBlur(img, (0, 0), 2).astype(np.float32) - ref, axis=2)
    cand = dist < 28
    lab, n = ndi.label(cand)
    edge_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    mask = np.isin(lab, edge_ids[edge_ids > 0])
    # 小さな穴を埋めて、縁をなめらかに
    mask = ndi.binary_fill_holes(mask) & cand | (mask & cand)
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3), ref


def finish(ai_rgb, aligned_dark, *, saturation=1.0, brightness=0.0, contrast=1.0, warmth=0.0,
           line_strength=0.8, line_color="#2B2624", paper="そのまま", paper_strength=1.0):
    """AI 着色画(RGB uint8)を仕上げて RGB uint8 で返す。"""
    img = ai_rgb.copy()

    # 背景の紙を統一
    if PAPERS[paper] is not None:
        mask, ref = _background_mask(img)
        h, w = img.shape[:2]
        if PAPERS[paper] == "gradient":
            yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
            target = np.array([252, 251, 249], np.float32) * (1 - yy) + np.array([236, 238, 242], np.float32) * yy
            target = np.broadcast_to(target, (h, w, 3))
        else:
            target = np.array(PAPERS[paper], np.float32)
        # 紙のざらつき（元の明暗のゆらぎ）は残す
        texture = (img.astype(np.float32) - ref[None, None, :]) * 0.5
        newbg = target + texture
        m = (mask * paper_strength)[..., None]
        img = np.clip(img * (1 - m) + newbg * m, 0, 255).astype(np.uint8)

    # 色調整
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = (out - 128) * contrast + 128 + brightness * 255
    out[..., 0] += warmth * 14
    out[..., 2] -= warmth * 14
    out = np.clip(out, 0, 255)

    # 元の線画を乗算で重ねる
    if line_strength > 0 and aligned_dark is not None:
        lc = hex_to_rgb(line_color) / 255.0
        alpha = np.clip(aligned_dark, 0, 1) * line_strength
        out = out * (1 - alpha[..., None] * (1 - lc[None, None, :]))
    return np.clip(out, 0, 255).astype(np.uint8)
