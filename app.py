"""パステル着色アプリ（Streamlit）。起動: streamlit run app.py

流れ: ① 線画を整える＋指示文を作る → ChatGPT/Gemini で着色 → ② 結果を取り込んで仕上げる
"""
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from align import align_lines  # noqa: E402
from finish import PAPERS, finish  # noqa: E402
from lineart import clean_lineart, decode_image, extract_lines  # noqa: E402
from prompt import BACKGROUNDS, DEFAULT_PARTS, DENSITY, SATURATION, TEXTURES, build_prompt  # noqa: E402

st.set_page_config(page_title="dMr_collarApp", layout="wide")
st.logo(str(Path(__file__).resolve().parent / "assets" / "logo.png"), size="large")
# タイトルの文字サイズは標準（約2.75rem）の70%
st.markdown("<h1 style='font-size:1.9rem; padding:0 0 0.5rem 0;'>dMr_collarApp</h1>", unsafe_allow_html=True)
st.caption("ChatGPT / Gemini での着色を、安定した品質に仕上げるための道具です。（API キー不要・無料）")


def to_png(rgb: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner="線画を解析しています…", max_entries=5, ttl=3600)
def prepare(data: bytes, sensitivity: float):
    bgr = decode_image(data, max_side=2000)
    dark = extract_lines(bgr, sensitivity)
    return clean_lineart(dark)


@st.cache_data(show_spinner="位置合わせしています…", max_entries=10, ttl=3600)
def align(orig_data: bytes, ai_data: bytes, sensitivity: float):
    orig = decode_image(orig_data, max_side=3000)
    ai = decode_image(ai_data, max_side=2000)
    dark, cc, aspect = align_lines(orig, ai, sensitivity)
    return ai[..., ::-1].copy(), dark, cc, aspect  # BGR → RGB


# ---------- 共通：元の線画 ----------
up = st.file_uploader("モノクロの線画（スマホ写真も可）", type=["png", "jpg", "jpeg"])

# タブは線画のアップロード前でも常に表示する
tab1, tab2 = st.tabs(["① 線画を整える・指示文を作る", "② 着色結果を取り込んで仕上げる"])
if up is None:
    for t in (tab1, tab2):
        with t:
            st.info("上のアップロード欄に、モノクロの線画をアップロードしてください。")
    st.stop()
orig_data = up.getvalue()

# =============== タブ1 ===============
with tab1:
    sensitivity = st.slider("線の拾いやすさ（薄い線が消えるときは小さく）", 0.6, 0.98, 0.90, 0.01)
    clean = prepare(orig_data, sensitivity)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("A. AI に渡す線画")
        st.image(clean, caption="紙の影を除去した線画", use_container_width=True)
        st.download_button("この線画を保存（PNG）", to_png(clean), file_name="lineart_clean.png", mime="image/png")
        st.caption("この画像を ChatGPT / Gemini に添付してください。")
    with c2:
        st.subheader("B. 着色の指示文")
        parts = {}
        with st.expander("配色（パーツごと）", expanded=True):
            for name, (desc, code) in DEFAULT_PARTS.items():
                a, b = st.columns([3, 1])
                d = a.text_input(name, desc, key=f"d_{name}")
                cc = b.color_picker("色", code, key=f"c_{name}", label_visibility="collapsed")
                parts[name] = (d, cc)
        s1, s2 = st.columns(2)
        sat = s1.select_slider("彩度", SATURATION, value="控えめ")
        den = s2.select_slider("塗りの濃さ", DENSITY, value="淡い")
        texture = st.selectbox("質感", list(TEXTURES.keys()))
        bg = st.selectbox("背景", BACKGROUNDS)
        extra = st.text_area("追加指示（任意）", placeholder="例: 髪にうっすら艶を入れる")
        prompt = build_prompt(parts, saturation=sat, density=den, texture=texture, background=bg, extra=extra)
        st.caption("右上のコピーボタンで指示文をコピーして、線画と一緒に ChatGPT / Gemini に送ります。")
        st.code(prompt, language="markdown", wrap_lines=True)

# =============== タブ2 ===============
with tab2:
    ups = st.file_uploader("ChatGPT / Gemini で着色した画像（複数可）", type=["png", "jpg", "jpeg", "webp"],
                           accept_multiple_files=True)
    if not ups:
        st.info("着色した画像をアップロードすると、元の線画を重ねて仕上げます。")
    else:
        left, right = st.columns([1, 3])
        with left:
            st.subheader("仕上げ")
            use_lines = st.checkbox("元の線画を重ねる", True)
            line_strength = st.slider("線の濃さ", 0.0, 1.0, 0.8, 0.05, disabled=not use_lines)
            line_color = st.color_picker("線の色", "#2B2624")
            saturation = st.slider("彩度", 0.5, 1.5, 1.0, 0.05)
            brightness = st.slider("明度", -0.2, 0.2, 0.0, 0.01)
            contrast = st.slider("コントラスト", 0.7, 1.3, 1.0, 0.05)
            warmth = st.slider("色温度（青←→黄）", -1.0, 1.0, 0.0, 0.1)
            paper = st.selectbox("背景の紙", list(PAPERS.keys()))
            paper_strength = st.slider("背景の置き換え強度", 0.0, 1.0, 1.0, 0.05, disabled=paper == "そのまま")
            compare = st.checkbox("並べて比較する", len(ups) > 1)

        results = []
        for f in ups:
            ai_rgb, dark, cc, aspect = align(orig_data, f.getvalue(), sensitivity)
            out = finish(ai_rgb, dark if use_lines else None, saturation=saturation, brightness=brightness,
                         contrast=contrast, warmth=warmth, line_strength=line_strength, line_color=line_color,
                         paper=paper, paper_strength=paper_strength)
            results.append((f.name, out, cc, aspect))

        with right:
            if compare and len(results) < 2:
                st.info("比較するには、着色した画像を2枚以上アップロードしてください（上のアップロード欄に追加できます）。")
            if compare and len(results) > 1:
                cols = st.columns(len(results))
                for col, (name, out, cc, aspect) in zip(cols, results):
                    col.image(out, caption=name, use_container_width=True)
                    col.download_button("PNG 保存", to_png(out), file_name=f"{Path(name).stem}_finished.png",
                                        mime="image/png", key=f"dl_{name}")
            else:
                names = [r[0] for r in results]
                pick = st.radio("表示する画像", names, horizontal=True) if len(names) > 1 else names[0]
                name, out, cc, aspect = next(r for r in results if r[0] == pick)
                st.image(out, caption=name, use_container_width=True)
                st.download_button("PNG 保存", to_png(out), file_name=f"{Path(name).stem}_finished.png",
                                   mime="image/png", key=f"dl1_{name}")
            for name, out, cc, aspect in results:
                if aspect > 0.03:
                    st.warning(f"{name}: 元の線画と縦横比が違います（{aspect:.0%}）。AI にトリミングされた可能性があり、線がずれることがあります。")
                if cc < 0.5:
                    st.warning(f"{name}: 線の位置合わせの一致度が低い（{cc:.2f}）ため、線の重ねが不自然な場合は『元の線画を重ねる』をオフにしてください。")
