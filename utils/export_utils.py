# utils/export_utils.py
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import streamlit as st
import textwrap
from datetime import datetime

def safe_bytes(data):
    return data.getvalue() if hasattr(data, "getvalue") else data

@st.cache_data(ttl=3600, show_spinner=False)
def df_to_png(df, title=None, width=1000, header=True, col_widths=None):
    dpi = 120
    row_count = max(1, len(df))
    height = 120 + row_count * 28 + (30 if header and title else 0)
    fig_w = width / dpi; fig_h = max(3.0, height / dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0,0,1,1])
    ax.set_facecolor('#0b0d10')
    y0 = 0.98
    if title:
        ax.text(0.01, y0, title, fontsize=16, fontweight='bold', color='white', va='top')
        y0 -= 0.05
    col_labels = df.columns.tolist()
    kwargs = {'cellText': df.values, 'colLabels': col_labels, 'cellLoc': 'left', 'loc': 'upper left', 'bbox': [0.01,0.01,0.98,y0-0.01]}
    if col_widths and len(col_widths)==len(col_labels):
        kwargs['colWidths'] = col_widths
    table = ax.table(**kwargs)
    table.auto_set_font_size(False); table.set_fontsize(11)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor('#222')
        if r==0:
            cell.set_facecolor('#14232b'); cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_text_props(color='white'); cell.set_facecolor('#0f1114' if r%2==0 else '#0b0d10')
    ax.axis('off')
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor()); plt.close(fig); buf.seek(0)
    return buf

def fig_to_png_bytes(fig, width=1000, height=600):
    try:
        img_bytes = fig.to_image(format='png', width=width, height=height, scale=2)
        return BytesIO(img_bytes)
    except Exception:
        return None

def clean_summary_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    txt = raw_text.replace("**", "").replace("\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\\n", " ").replace("\\r", " ")
    txt = txt.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    txt = txt.replace("", " ").strip()
    parts = txt.split()
    for i, p in enumerate(parts):
        try:
            if "-" in p and len(p.split("-")) == 3:
                d = datetime.strptime(p, "%Y-%m-%d").strftime("%d/%m/%Y")
                parts[i] = d
        except Exception:
            pass
    txt = " ".join(parts).replace("  ", " ").replace(" :", ":")
    return txt.strip()

def build_season_review_card(sections, width=1080):
    card_bg = (11, 13, 16)
    pad_x, pad_y = 40, 40
    section_spacing = 50
    section_images = []
    width = int(width)
    try:
        font_title = ImageFont.truetype("arial.ttf", 38)
        font_section = ImageFont.truetype("arial.ttf", 30)
        font_body = ImageFont.truetype("arial.ttf", 22)
    except:
        font_title = font_section = font_body = ImageFont.load_default()

    header_h = 160
    header = Image.new("RGB", (width, header_h), card_bg)
    draw = ImageDraw.Draw(header)
    for x in range(width):
        r = int(46 + (214 - 46) * (x / width))
        g = int(134 + (69 - 134) * (x / width))
        b = int(171 + (69 - 171) * (x / width))
        draw.line([(x, 0), (x, header_h)], fill=(r, g, b))
    title_text = "Love Five-A-Side — Season Review"
    tw, th = draw.textbbox((0, 0), title_text, font=font_title)[2:]
    draw.text(((width - tw) // 2, (header_h - th) // 2), title_text, font=font_title, fill="white")
    section_images.append(header)

    def _text_block(text):
        txt = clean_summary_text(text)
        lines = textwrap.wrap(txt, width=90)
        from PIL import Image as _Img
        text_img_h = 100 + 28 * len(lines)
        text_img = _Img.new("RGB", (width, text_img_h), card_bg)
        d = ImageDraw.Draw(text_img)
        y = 40
        for line in lines:
            d.text((pad_x, y), line, font=font_body, fill=(230, 238, 246))
            y += 28
        return text_img

    for title, block in sections:
        section_h = 70
        sec_img = Image.new("RGB", (width, section_h), (15, 17, 20))
        d = ImageDraw.Draw(sec_img)
        d.text((30, 18), title, font=font_section, fill=(230, 238, 246))
        section_images.append(sec_img)

        if isinstance(block, str):
            section_images.append(_text_block(block))
        elif hasattr(block, "columns"):
            tbl_buf = df_to_png(block, title=None, width=width - 2 * pad_x)
            from PIL import Image as _Img
            tbl = _Img.open(BytesIO(tbl_buf.getvalue())).convert("RGB")
            w, h = tbl.size
            bg = _Img.new("RGB", (width, h + pad_y * 2), card_bg)
            bg.paste(tbl, (pad_x, pad_y))
            section_images.append(bg)
        else:
            try:
                from PIL import Image as _Img
                tbl = _Img.open(BytesIO(block)).convert("RGB")
                w, h = tbl.size
                bg = _Img.new("RGB", (width, h + pad_y * 2), card_bg)
                bg.paste(tbl, (pad_x, pad_y))
                section_images.append(bg)
            except Exception:
                section_images.append(_text_block(str(block)))

        from PIL import Image as _Img
        section_images.append(_Img.new("RGB", (width, section_spacing), card_bg))

    footer_h = 80
    from PIL import Image as _Img
    footer = _Img.new("RGB", (width, footer_h), (15, 17, 20))
    d = ImageDraw.Draw(footer)
    txt = f"Generated in LoveFiveASide App • {datetime.today().strftime('%d/%m/%Y')}"
    tw, th = d.textbbox((0, 0), txt, font=font_body)[2:]
    d.text(((width - tw) // 2, (footer_h - th) // 2), txt, font=font_body, fill=(160, 170, 180))
    section_images.append(footer)

    total_h = sum(img.height for img in section_images)
    final_img = _Img.new("RGB", (width, total_h), card_bg)
    y_offset = 0
    for img in section_images:
        final_img.paste(img, (0, y_offset)); y_offset += img.height
    out = BytesIO(); final_img.save(out, format="PNG", optimize=True); out.seek(0)
    return out
