# -*- coding: utf-8 -*-
"""绘制三阶段微调流程图，去重、紧凑布局，输出 PNG。仅依赖 Pillow。"""
from PIL import Image, ImageDraw, ImageFont

def main():
    # 紧凑画布：宽约 900，高约 220，留白小
    w, h = 900, 220
    img = Image.new('RGB', (w, h), color='white')
    draw = ImageDraw.Draw(img)

    # 尝试加载中文字体（Windows）
    import os
    font_paths = [
        os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'msyh.ttc'),
        os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'simhei.ttf'),
        'msyh.ttc', 'simhei.ttf'
    ]
    font_title = font_text = None
    for fp in font_paths:
        if os.path.isfile(fp):
            try:
                font_title = ImageFont.truetype(fp, 18)
                font_text = ImageFont.truetype(fp, 14)
                break
            except Exception:
                pass
    if font_title is None:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    box_w, box_h = 240, 120
    gap = 40
    y_center = h // 2
    x_start = 30

    def round_rect(d, xy, r, fill, outline):
        x1, y1, x2, y2 = xy
        d.rectangle([x1 + r, y1, x2 - r, y2], fill=fill)
        d.rectangle([x1, y1 + r, x2, y2 - r], fill=fill)
        d.pieslice([x1, y1, x1 + 2*r, y1 + 2*r], 180, 270, fill=fill, outline=outline)
        d.pieslice([x2 - 2*r, y1, x2, y1 + 2*r], 270, 360, fill=fill, outline=outline)
        d.pieslice([x1, y2 - 2*r, x1 + 2*r, y2], 90, 180, fill=fill, outline=outline)
        d.pieslice([x2 - 2*r, y2 - 2*r, x2, y2], 0, 90, fill=fill, outline=outline)
        d.rectangle([x1, y1 + r, x1 + r, y2 - r], fill=fill)
        d.rectangle([x2 - r, y1 + r, x2, y2 - r], fill=fill)
        d.rectangle([x1 + r, y1, x2 - r, y1 + r], fill=fill)
        d.rectangle([x1 + r, y2 - r, x2 - r, y2], fill=fill)
        d.rounded_rectangle(xy, radius=r, outline=outline, width=2)

    r = 12
    blue_fill, blue_outline = '#B3D9FF', '#3399FF'
    green_fill, green_outline = '#B8E6B8', '#2E7D32'

    # 阶段一
    x1 = x_start
    rect1 = (x1, y_center - box_h//2, x1 + box_w, y_center + box_h//2)
    draw.rounded_rectangle(rect1, radius=r, fill=blue_fill, outline=blue_outline, width=2)
    draw.text((x1 + box_w//2, y_center - box_h//2 + 18), '阶段一：模态对齐', fill='#000', font=font_title, anchor='mm')
    draw.text((x1 + box_w//2, y_center - 2), '冻结 Encoder，解冻 Projector，冻结 LLM', fill='#333', font=font_text, anchor='mm')
    draw.text((x1 + box_w - 38, y_center + 28), '锁', fill=blue_outline, font=font_text, anchor='mm')

    # 箭头 1->2
    ax1, ax2 = x1 + box_w + 8, x1 + box_w + gap - 8
    draw.line([(ax1, y_center), (ax2 - 12, y_center)], fill=blue_outline, width=2)
    draw.polygon([(ax2 - 12, y_center), (ax2 - 22, y_center - 8), (ax2 - 22, y_center + 8)], fill=blue_outline)

    # 阶段二
    x2 = x1 + box_w + gap
    rect2 = (x2, y_center - box_h//2, x2 + box_w, y_center + box_h//2)
    draw.rounded_rectangle(rect2, radius=r, fill=blue_fill, outline=blue_outline, width=2)
    draw.text((x2 + box_w//2, y_center - box_h//2 + 18), '阶段二：指令微调', fill='#000', font=font_title, anchor='mm')
    draw.text((x2 + box_w//2, y_center - 2), '冻结 Encoder，部分解冻 Projector+LLM', fill='#333', font=font_text, anchor='mm')
    draw.text((x2 + box_w - 38, y_center + 28), '锁', fill=blue_outline, font=font_text, anchor='mm')

    # 箭头 2->3
    bx1, bx2 = x2 + box_w + 8, x2 + box_w + gap - 8
    draw.line([(bx1, y_center), (bx2 - 12, y_center)], fill=blue_outline, width=2)
    draw.polygon([(bx2 - 12, y_center), (bx2 - 22, y_center - 8), (bx2 - 22, y_center + 8)], fill=blue_outline)

    # 阶段三
    x3 = x2 + box_w + gap
    rect3 = (x3, y_center - box_h//2, x3 + box_w, y_center + box_h//2)
    draw.rounded_rectangle(rect3, radius=r, fill=green_fill, outline=green_outline, width=2)
    draw.text((x3 + box_w//2, y_center - box_h//2 + 18), '阶段三：生成微调', fill='#000', font=font_title, anchor='mm')
    draw.text((x3 + box_w//2, y_center - 2), '解冻 Projector+Generator，冻结或微调 LLM', fill='#333', font=font_text, anchor='mm')
    draw.text((x3 + box_w - 38, y_center + 28), '开', fill=green_outline, font=font_text, anchor='mm')

    out_path = r'C:\Users\fanji\Desktop\nudt_thesis\figures\fig_three_stage_optimized.png'
    img.save(out_path)
    print('Saved:', out_path)

if __name__ == '__main__':
    main()
