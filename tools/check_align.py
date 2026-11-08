"""AI 着色画に元の線画を位置合わせして重ねられるかの検証。
使い方: python tools/check_align.py <元の線画> <AI着色画> <出力フォルダ>"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lineart import decode_image, extract_lines  # noqa: E402


def save(path, arr):
    ok, buf = cv2.imencode(".png", arr)
    buf.tofile(str(path))


orig_p, ai_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

ai = decode_image(ai_p.read_bytes(), max_side=1600)
h, w = ai.shape[:2]
orig = decode_image(orig_p.read_bytes(), max_side=3000)
print("元:", orig.shape[:2], "AI:", (h, w), "縦横比 元/AI:",
      round(orig.shape[1] / orig.shape[0], 3), round(w / h, 3))

# 元の線画（紙の影を除去した線の濃さ）
d_orig = extract_lines(orig)
# AI 画像の線の濃さ（色の付いた面を線と誤認しないよう、彩度の低い暗部だけ見る）
hsv = cv2.cvtColor(ai, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(ai, cv2.COLOR_BGR2GRAY).astype(np.float32)
bg = cv2.medianBlur(cv2.cvtColor(ai, cv2.COLOR_BGR2GRAY), 51).astype(np.float32)
d_ai = np.clip((1 - gray / np.maximum(bg, 1)) * 2.5 * (1 - hsv[..., 1] / 255.0 * 0.7), 0, 1)

# 元の線画を AI 画像の大きさへ（縦横比が違う場合は中央を合わせてフィット）
d_o = cv2.resize(d_orig, (w, h), interpolation=cv2.INTER_AREA)
a = cv2.GaussianBlur(d_ai, (0, 0), 3).astype(np.float32)
b = cv2.GaussianBlur(d_o, (0, 0), 3).astype(np.float32)

warp = np.eye(3, dtype=np.float32)
cc = 0.0
for sc in (0.25, 0.5, 1.0):  # 粗い解像度から順に合わせる
    sa = cv2.resize(a, None, fx=sc, fy=sc)
    sb = cv2.resize(b, None, fx=sc, fy=sc)
    wm = warp.copy()
    wm[:2, 2] *= sc
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        cc, wm = cv2.findTransformECC(sa, sb, wm, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    except cv2.error as e:
        print("ECC 失敗 scale", sc, str(e)[:80])
        break
    wm[:2, 2] /= sc
    warp = wm
print("ECC 相関:", round(float(cc), 3))

aligned = cv2.warpPerspective(d_o, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# 重ね合わせ: AI の線 → 青、元の線 → 赤（重なれば紫〜黒）で位置ずれを可視化
vis = np.full((h, w, 3), 255, np.uint8)
vis[..., 0] = (255 * (1 - np.clip(d_ai * 1.2, 0, 1))).astype(np.uint8)  # B を減らす=赤味
vis[..., 1] = (255 * (1 - np.clip(d_ai * 1.2, 0, 1)) * (1 - np.clip(aligned * 1.2, 0, 1))).astype(np.uint8)
vis[..., 2] = (255 * (1 - np.clip(aligned * 1.2, 0, 1))).astype(np.uint8)
save(out / "align_diff.png", vis)

# 元の線を乗算で重ねた結果
res = ai.astype(np.float32) * (1 - np.clip(aligned, 0, 1)[..., None] * 0.9 * (1 - 0.15))
save(out / "align_result.png", np.clip(res, 0, 255).astype(np.uint8))
save(out / "ai_original.png", ai)
