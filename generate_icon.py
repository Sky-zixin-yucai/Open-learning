#!/usr/bin/env python3
import os
import sys
import math
import random
import argparse
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps
import numpy as np

# ── 输出路径 ──────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
PNG_PATH = os.path.join(ASSETS_DIR, "icon.png")
ICO_PATH = os.path.join(ASSETS_DIR, "icon.ico")
ICNS_PATH = os.path.join(ASSETS_DIR, "icon.icns")

# ── 默认配置 ──────────────────────────────────────────
DEFAULT_CONFIG = {
    "ssaa": 2,                     # 超采样倍数
    "light_azimuth": -0.45,        # 光源水平方向 (-1~1)
    "light_elevation": -0.55,      # 光源垂直方向 (-1~1)
    "petal_aspect": 2.4,           # 花瓣长宽比
    "notch_depth": 0.18,           # 花瓣缺口深度
    "stamen_count": 14,            # 花蕊数量
    "anther_size": 0.009,          # 花药相对尺寸
    "vignette_radius": 0.72,       # 暗角半径
    "texture_intensity": 0.02,     # 肌理强度
    "simplify_threshold": 128,     # 小于此尺寸使用简化渲染
}

# ── 配色方案（优化颜色值，提升对比度） ──────────────
COLOR_THEMES = {
    "sakura": {
        "bg_top": (86, 58, 128),
        "bg_mid": (108, 76, 158),
        "bg_bottom": (138, 102, 186),
        "petals": [
            (255, 172, 195), (255, 166, 190), (255, 178, 200),
            (255, 169, 193), (255, 175, 198),
        ],
        "petal_edge": (245, 140, 170),
        "petal_shadow": (255, 172, 195),
        "petal_vein": (220, 130, 160),
        "center_gold": (255, 190, 45),
        "center_core": (245, 155, 0),
        "stamen": (255, 180, 40),
        "stamen_tip": (255, 210, 80),
        "spotlight": (255, 240, 255),
        "vignette": (30, 15, 55),
    },
    "night": {
        "bg_top": (22, 22, 42),
        "bg_mid": (38, 33, 68),
        "bg_bottom": (58, 48, 92),
        "petals": [
            (200, 180, 240), (190, 170, 235), (210, 190, 245),
            (195, 175, 238), (205, 185, 242),
        ],
        "petal_edge": (160, 130, 210),
        "petal_shadow": (100, 80, 160),
        "petal_vein": (170, 150, 220),
        "center_gold": (255, 200, 60),
        "center_core": (230, 140, 0),
        "stamen": (255, 190, 50),
        "stamen_tip": (255, 220, 90),
        "spotlight": (200, 200, 255),
        "vignette": (10, 8, 25),
    },
    "dawn": {
        "bg_top": (175, 95, 115),
        "bg_mid": (205, 125, 135),
        "bg_bottom": (235, 175, 165),
        "petals": [
            (255, 220, 225), (255, 215, 222), (255, 225, 228),
            (255, 218, 224), (255, 222, 226),
        ],
        "petal_edge": (225, 150, 165),
        "petal_shadow": (190, 120, 140),
        "petal_vein": (220, 155, 170),
        "center_gold": (255, 180, 50),
        "center_core": (240, 140, 0),
        "stamen": (255, 170, 40),
        "stamen_tip": (255, 205, 75),
        "spotlight": (255, 235, 230),
        "vignette": (80, 40, 50),
    },
    "ocean": {
        "bg_top": (25, 70, 110),
        "bg_mid": (40, 100, 145),
        "bg_bottom": (70, 140, 180),
        "petals": [
            (200, 230, 245), (190, 222, 240), (210, 235, 248),
            (195, 226, 243), (205, 230, 246),
        ],
        "petal_edge": (120, 175, 210),
        "petal_shadow": (80, 130, 175),
        "petal_vein": (130, 185, 215),
        "center_gold": (255, 200, 70),
        "center_core": (235, 150, 10),
        "stamen": (255, 185, 45),
        "stamen_tip": (255, 215, 85),
        "spotlight": (220, 240, 255),
        "vignette": (10, 35, 60),
    },
    "forest": {
        "bg_top": (45, 80, 55),
        "bg_mid": (65, 110, 75),
        "bg_bottom": (95, 140, 100),
        "petals": [
            (230, 240, 220), (222, 235, 215), (235, 243, 225),
            (226, 238, 218), (230, 240, 222),
        ],
        "petal_edge": (150, 185, 130),
        "petal_shadow": (100, 140, 95),
        "petal_vein": (155, 190, 135),
        "center_gold": (255, 195, 50),
        "center_core": (240, 150, 0),
        "stamen": (255, 175, 40),
        "stamen_tip": (255, 210, 80),
        "spotlight": (235, 250, 230),
        "vignette": (15, 35, 20),
    },
    "midnight": {
        "bg_top": (12, 12, 18),
        "bg_mid": (22, 22, 32),
        "bg_bottom": (30, 30, 42),
        "petals": [
            (245, 245, 250), (238, 238, 245), (250, 250, 252),
            (242, 242, 248), (247, 247, 250),
        ],
        "petal_edge": (200, 200, 215),
        "petal_shadow": (180, 180, 200),
        "petal_vein": (200, 200, 215),
        "center_gold": (255, 210, 120),
        "center_core": (240, 180, 80),
        "stamen": (255, 220, 140),
        "stamen_tip": (255, 235, 180),
        "spotlight": (255, 255, 255),
        "vignette": (0, 0, 0),
    },
}

# ══════════════════════════════════════════════════════
#  颜色与几何工具
# ══════════════════════════════════════════════════════

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, int(v)))

def lerp(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(clamp(c1[i] + (c2[i] - c1[i]) * t) for i in range(len(c1)))

def rotate_point(x, y, angle):
    ca, sa = math.cos(angle), math.sin(angle)
    return x * ca - y * sa, x * sa + y * ca

def smoothstep(edge0, edge1, x):
    """
    平滑插值函数，支持标量及 NumPy 数组。
    """
    if isinstance(x, np.ndarray):
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3 - 2 * t)
    else:
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3 - 2 * t)

def bezier_quad(p0, p1, p2, t):
    mt = 1 - t
    return (mt*mt*p0[0] + 2*mt*t*p1[0] + t*t*p2[0],
            mt*mt*p0[1] + 2*mt*t*p1[1] + t*t*p2[1])

# ══════════════════════════════════════════════════════
#  渲染器类
# ══════════════════════════════════════════════════════

class IconRenderer:
    """樱花图标渲染器"""

    def __init__(self, size=1024, theme="midnight", config=None):
        self.size = size
        self.theme_name = theme
        self.theme = COLOR_THEMES[theme]
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.ssaa = self.config["ssaa"]
        self.big = size * self.ssaa
        self.radius_big = int(self.big * 0.225)
        self.light_dir = (self.config["light_azimuth"], self.config["light_elevation"])
        self.rng = random.Random(42)  # 固定种子

        # 简化渲染标志
        self.simplify = size <= self.config["simplify_threshold"]

    # ---------- 主入口 ----------
    def render(self):
        """执行渲染，返回 PIL.Image 对象（已缩放到目标尺寸）

        满框正方形，macOS 自动裁切为 squircle 形状。
        """
        img = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))

        self._draw_background(img)
        self._draw_flower(img)
        if not self.simplify:
            self._draw_sparkles(img)

        # 合成到纯不透明背景，消除 paste 导致的半透明像素（macOS 图标要求满框不透明）
        flat = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 255))
        flat.alpha_composite(img)
        return flat.resize((self.size, self.size), Image.LANCZOS)

    # ---------- 背景 ----------
    def _draw_background(self, img):
        """深紫黑径向渐变背景，中心略亮，边缘渐暗，避免纯黑方块感。"""
        import numpy as np
        size = self.big
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        cx, cy = size / 2, size / 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size / 2)
        dist = np.clip(dist, 0, 1)
        # 中心深紫 (#1a0a2e) → 边缘纯黑 (#000)
        r = (26 * (1 - dist) ** 1.5).astype(np.uint8)
        g = (10 * (1 - dist) ** 1.5).astype(np.uint8)
        b = (46 * (1 - dist) ** 1.5).astype(np.uint8)
        arr[:, :, 0] = r
        arr[:, :, 1] = g
        arr[:, :, 2] = b
        arr[:, :, 3] = 255
        bg = Image.fromarray(arr, "RGBA")
        img.alpha_composite(bg)

    def _make_diagonal_gradient(self, size, c1, c2, c3):
        """三段对角渐变（小图放大，平滑无条带）"""
        s = 256
        small = Image.new("RGBA", (s, s))
        sd = small.load()
        for y in range(s):
            for x in range(s):
                t = (x / s) * 0.45 + (y / s) * 0.55
                if t < 0.5:
                    c = lerp(c1, c2, t * 2)
                else:
                    c = lerp(c2, c3, (t - 0.5) * 2)
                sd[x, y] = (*c, 255)
        return small.resize((size, size), Image.BICUBIC)

    def _add_radial_spot(self, img, center, radius, color, alpha=32):
        """在图像上叠加径向渐变光斑（只衰减alpha，不褪色到白色）"""
        cx, cy = int(center[0] * self.big), int(center[1] * self.big)
        r = int(radius * self.big)
        x = np.arange(self.big, dtype=np.float64)
        y = np.arange(self.big, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        t = np.clip(d / r, 0, 1)
        arr = np.zeros((self.big, self.big, 4), dtype=np.uint8)
        arr[:, :, 0] = color[0]
        arr[:, :, 1] = color[1]
        arr[:, :, 2] = color[2]
        arr[:, :, 3] = (alpha * (1 - t) ** 1.5).astype(np.uint8)
        spot = Image.fromarray(arr, "RGBA")
        spot = spot.filter(ImageFilter.GaussianBlur(r * 0.06))
        img.alpha_composite(spot)

    def _make_vignette(self, size, color, radius_ratio):
        """创建暗角图层（使用smoothstep柔化）"""
        cx, cy = size / 2, size / 2
        outer = radius_ratio * size / 2
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        for i in range(3):
            arr[:, :, i] = color[i]
        y, x = np.ogrid[:size, :size]
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        t = smoothstep(outer * 0.6, outer, d)
        arr[:, :, 3] = (t * 65).astype(np.uint8)
        vignette = Image.fromarray(arr, "RGBA")
        return vignette

    # ---------- 花朵 ----------
    def _draw_flower(self, img):
        cx, cy = self.big / 2, self.big / 2
        base_len = self.big * 0.28
        base_w = base_len / 2.1

        # 多层花瓣从外到内、从大到小逐层绽放
        # off_ratio≈1.0 时花瓣基部恰好在中心，不越过中心造成重叠
        layers = [
            # (大小比例, 角度偏移, 距中心偏移比例, 提亮, 立体强度)
            (1.00,   0, 0.94, 1.00, 1.00),   # 外层
            (0.92,  36, 0.96, 1.03, 0.55),   # 中层，错开36°
            (0.84,   0, 0.98, 1.06, 0.25),   # 内层，基部往中心收一点盖住洞
        ]
        for scale, ang_offset, off_ratio, brighten, depth in layers:
            plen = base_len * scale
            pwid = base_w * scale
            off = plen * off_ratio
            for i in range(5):
                radial = math.radians(-90 + ang_offset + i * 72)
                angle = radial + math.pi / 2
                px = cx + math.cos(radial) * off
                py = cy + math.sin(radial) * off
                base_color = self.theme["petals"][i]
                color = tuple(min(255, int(c * brighten)) for c in base_color)
                self._draw_petal_body(img, px, py, pwid, plen, angle, color, i, depth)

    # ---------- 花瓣 ----------
    def _petal_outline(self, cx, cy, rx, ry, angle):
        """樱花花瓣轮廓：基部窄、中部宽、顶端心形缺口（正确闭合多边形）"""
        points = []
        steps = 120
        # 右半：tip(t=0) → base(t=π)
        for i in range(steps + 1):
            t = (i / steps) * math.pi
            env = math.sin(t) ** 0.68
            x = rx * env
            y = -ry * math.cos(t)
            if t < 0.28:
                depth = self.config["notch_depth"] * ry * (1 - t / 0.28) ** 2
                y += depth
            wave = math.sin(t * 2.8) * rx * 0.010
            x += wave * math.sin(t)
            px, py = rotate_point(x, y, angle)
            points.append((cx + px, cy + py))
        # 左半反向：base(t=π) → tip(t=0)，避免多边形自交
        for i in range(steps, -1, -1):
            t = (i / steps) * math.pi
            env = math.sin(t) ** 0.68
            x = -rx * env
            y = -ry * math.cos(t)
            if t < 0.28:
                depth = self.config["notch_depth"] * ry * (1 - t / 0.28) ** 2
                y += depth
            wave = math.sin(t * 2.8) * rx * 0.010
            x -= wave * math.sin(t)
            px, py = rotate_point(x, y, angle)
            points.append((cx + px, cy + py))
        return points

    def _draw_petal_body(self, img, cx, cy, rx, ry, angle, color, idx, depth=1.0):
        """绘制花瓣本体（无纹路无阴影）"""
        points = self._petal_outline(cx, cy, rx, ry, angle)

        # 底色 + 边缘
        self._draw_petal_base(img, points, color)
        # 内部渐变（立体效果，depth控制强度）
        self._draw_petal_gradient(img, cx, cy, rx, ry, angle, color, depth)
        # 纹理
        if not self.simplify:
            self._draw_petal_texture(img, points)

    def _draw_petal_shadow(self, img, points):
        """花瓣投影（光源左上，阴影偏右下）"""
        shadow = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        # 阴影方向与光源相反
        dx = -self.light_dir[0] * self.big * 0.008
        dy = -self.light_dir[1] * self.big * 0.008
        shadow_pts = [(p[0] + dx, p[1] + dy) for p in points]
        sd.polygon(shadow_pts, fill=(45, 20, 75, 40))
        shadow = shadow.filter(ImageFilter.GaussianBlur(self.big * 0.010))
        img.alpha_composite(shadow)

    def _draw_petal_base(self, img, points, color):
        """花瓣底色 + 半透明边缘"""
        draw = ImageDraw.Draw(img)
        edge = lerp(color, self.theme["petal_edge"], 0.30)
        draw.polygon(points, fill=(*color, 242), outline=(*edge, 120))

    def _draw_petal_gradient(self, img, cx, cy, rx, ry, angle, color, depth=1.0):
        """为每片花瓣单独计算立体渐变：中央脊线亮、两侧暗，depth控制强度"""
        points = self._petal_outline(cx, cy, rx, ry, angle)
        mask = Image.new("L", (self.big, self.big), 0)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(1, self.big // 900)))

        ca, sa = math.cos(angle), math.sin(angle)

        # 用numpy为这片花瓣生成专属立体渐变
        # 取花瓣包围盒
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, x1 = int(max(0, min(xs) - 4)), int(min(self.big, max(xs) + 4))
        y0, y1 = int(max(0, min(ys) - 4)), int(min(self.big, max(ys) + 4))
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return

        # 生成包围盒内的坐标网格
        gx = np.arange(x0, x1, dtype=np.float64)
        gy = np.arange(y0, y1, dtype=np.float64)
        xx, yy = np.meshgrid(gx, gy)

        # 反旋转到花瓣局部坐标
        dx = xx - cx
        dy = yy - cy
        lx = dx * ca + dy * sa
        ly = -dx * sa + dy * ca

        # 沿宽度方向的归一化距离 (0=中央脊线, 1=边缘)
        edge_t = np.clip(np.abs(lx) / rx, 0, 1)
        # 沿长度方向的位置 (0=基部, 1=尖端)
        len_t = np.clip((-ly / ry + 1) / 2, 0, 1)
        # 花瓣形状包络（只在花瓣内部生效）
        env = np.clip(1.0 - edge_t ** 2, 0, 1) ** 0.5

        # ── 中央脊线高光：高斯分布，沿长度方向中部最强 ──
        ridge_strength = np.exp(-(edge_t ** 2) / 0.12) * (np.sin(len_t * math.pi) ** 0.6)
        ridge_alpha = (ridge_strength * 30 * env * depth).astype(np.uint8)

        # ── 两侧暗影：边缘暗、中央亮，模拟卷曲 ──
        shade_curve = edge_t ** 1.8
        shade_strength = shade_curve * (0.6 + 0.4 * np.sin(len_t * math.pi))
        shade_alpha = (shade_strength * 18 * env * depth).astype(np.uint8)

        # ── 顶端额外亮光 ──
        tip_glow = np.exp(-((len_t - 0.72) ** 2) / 0.08) * np.exp(-(edge_t ** 2) / 0.30)
        tip_alpha = (tip_glow * 18 * env * depth).astype(np.uint8)

        # 合成RGBA层
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        # 高光（白色）
        layer[:, :, 0] = np.where(ridge_alpha > 0, 255, 0)
        layer[:, :, 1] = np.where(ridge_alpha > 0, 240, 0)
        layer[:, :, 2] = np.where(ridge_alpha > 0, 248, 0)
        layer[:, :, 3] = ridge_alpha
        # 叠加顶端亮光
        layer[:, :, 0] = np.maximum(layer[:, :, 0], np.where(tip_alpha > 0, 255, 0))
        layer[:, :, 1] = np.maximum(layer[:, :, 1], np.where(tip_alpha > 0, 245, 0))
        layer[:, :, 2] = np.maximum(layer[:, :, 2], np.where(tip_alpha > 0, 250, 0))
        layer[:, :, 3] = np.maximum(layer[:, :, 3], tip_alpha)

        # 暗影（比花瓣底色深一些，自动适配任意花色）
        shade_color = tuple(max(0, int(c * 0.72)) for c in color)
        shade_layer = np.zeros((h, w, 4), dtype=np.uint8)
        shade_layer[:, :, 0] = np.where(shade_alpha > 0, shade_color[0], 0)
        shade_layer[:, :, 1] = np.where(shade_alpha > 0, shade_color[1], 0)
        shade_layer[:, :, 2] = np.where(shade_alpha > 0, shade_color[2], 0)
        shade_layer[:, :, 3] = shade_alpha

        # 裁剪到花瓣蒙版
        sub_mask = np.array(mask)[y0:y1, x0:x1]
        layer[:, :, 3] = (layer[:, :, 3].astype(np.float64) * sub_mask / 255).astype(np.uint8)
        shade_layer[:, :, 3] = (shade_layer[:, :, 3].astype(np.float64) * sub_mask / 255).astype(np.uint8)

        # 合成到主图
        img.paste(Image.fromarray(layer, "RGBA"), (x0, y0),
                  Image.fromarray(layer[:, :, 3], "L"))
        shade_img = Image.fromarray(shade_layer, "RGBA")
        img.alpha_composite(shade_img, (x0, y0))

        # ── 边缘透光 ──
        glow = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.line(points + [points[0]], fill=(255, 240, 250, int(14 * depth)),
                width=max(2, int(rx * 0.10)))
        glow = glow.filter(ImageFilter.GaussianBlur(rx * 0.08))
        clipped_g = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        clipped_g.paste(glow, (0, 0), mask)
        img.alpha_composite(clipped_g)

    def _draw_petal_veins(self, img, cx, cy, rx, ry, angle):
        """贝塞尔曲线绘制脉络（主脉+侧脉），全部在花瓣局部坐标系计算后旋转"""
        draw = ImageDraw.Draw(img)
        ca, sa = math.cos(angle), math.sin(angle)
        theme = self.theme

        def _to_world(lx, ly):
            """局部坐标（花瓣轴向）→ 世界坐标"""
            wx, wy = rotate_point(lx, ly, angle)
            return cx + wx, cy + wy

        # 主脉
        pts = [_to_world(0, -ry * t) for t in np.linspace(0.05, 0.92, 20)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(*theme["petal_vein"], 55),
                      width=max(1, int(self.big * 0.0020)))

        # 侧脉（左右各3条，在局部坐标系计算贝塞尔曲线后旋转）
        for side in (-1, 1):
            for pos in (0.25, 0.45, 0.65):
                # 起点在主脉上
                p0 = (0, -ry * pos)
                # 终点向外上方延伸
                p2 = (side * rx * 0.55, -ry * (pos - 0.12))
                # 控制点
                p1 = (side * rx * 0.30, -ry * (pos - 0.04))
                vein_pts = []
                for t in np.linspace(0, 1, 10):
                    lx, ly = bezier_quad(p0, p1, p2, t)
                    vein_pts.append(_to_world(lx, ly))
                for i in range(len(vein_pts) - 1):
                    draw.line([vein_pts[i], vein_pts[i + 1]],
                              fill=(*theme["petal_vein"], 40),
                              width=max(1, int(self.big * 0.0014)))

    def _draw_petal_texture(self, img, points):
        """在花瓣内部添加细微纹理（numpy向量化）"""
        mask = Image.new("L", (self.big, self.big), 0)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(2))
        count = 40
        noise = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        nd = ImageDraw.Draw(noise)
        for _ in range(count):
            x = self.rng.randint(0, self.big - 1)
            y = self.rng.randint(0, self.big - 1)
            l = self.rng.randint(2, 5)
            a = self.rng.uniform(0, math.pi)
            nd.line([(x, y), (x + math.cos(a) * l, y + math.sin(a) * l)],
                    fill=(220, 200, 210, self.rng.randint(5, 12)), width=1)
        noise = noise.filter(ImageFilter.GaussianBlur(1.2))
        clipped = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        clipped.paste(noise, (0, 0), mask)
        img.alpha_composite(clipped)

    def _draw_notch_shadow(self, img, cx, cy, rx, ry, angle):
        """花瓣顶端缺口阴影"""
        layer = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        ca, sa = math.cos(angle), math.sin(angle)
        tip_x = cx + ca * (-ry * 0.88)
        tip_y = cy + sa * (-ry * 0.88)
        d.ellipse([tip_x - rx * 0.10, tip_y - ry * 0.035,
                   tip_x + rx * 0.10, tip_y + ry * 0.035],
                  fill=(*self.theme["petal_shadow"], 28))
        layer = layer.filter(ImageFilter.GaussianBlur(rx * 0.05))
        img.alpha_composite(layer)

    def _draw_inter_petal_shadow(self, img, cx, cy, rx, ry, angle):
        """花瓣间交叠阴影"""
        layer = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        ca, sa = math.cos(angle), math.sin(angle)
        sh_x = cx + ca * ry * 0.12
        sh_y = cy + sa * ry * 0.12
        d.ellipse([sh_x - rx * 0.55, sh_y - ry * 0.18,
                   sh_x + rx * 0.55, sh_y + ry * 0.18],
                  fill=(50, 25, 80, 30))
        layer = layer.filter(ImageFilter.GaussianBlur(rx * 0.18))
        img.alpha_composite(layer)

    # ---------- 花蕊 ----------
    def _draw_center_disc(self, img, cx, cy):
        """花蕊圆盘：简洁三层金色 + 柔光高光"""
        theme = self.theme
        r_outer = self.big * 0.066
        r_mid = self.big * 0.048
        r_core = self.big * 0.032

        # 投影
        shadow = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.ellipse([cx - r_outer + 2, cy - r_outer + 3,
                    cx + r_outer + 2, cy + r_outer + 3],
                   fill=(100, 60, 10, 45))
        shadow = shadow.filter(ImageFilter.GaussianBlur(self.big * 0.007))
        img.alpha_composite(shadow)

        d = ImageDraw.Draw(img)
        d.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
                  fill=(*theme["center_gold"], 255))
        d.ellipse([cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid],
                  fill=(*lerp(theme["center_gold"], theme["center_core"], 0.5), 255))
        d.ellipse([cx - r_core, cy - r_core, cx + r_core, cy + r_core],
                  fill=(*theme["center_core"], 255))
        # 高光
        hl = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        hd = ImageDraw.Draw(hl)
        hx, hy = cx - r_outer * 0.22, cy - r_outer * 0.28
        hr = r_core * 0.45
        hd.ellipse([hx - hr, hy - hr, hx + hr, hy + hr],
                   fill=(255, 245, 200, 160))
        hl = hl.filter(ImageFilter.GaussianBlur(r_outer * 0.08))
        img.alpha_composite(hl)

    def _draw_center_disc_simplified(self, img, cx, cy):
        """简化版花蕊（仅圆盘）"""
        r = self.big * 0.060
        d = ImageDraw.Draw(img)
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  fill=(*self.theme["center_gold"], 255))
        r2 = self.big * 0.032
        d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2],
                  fill=(*self.theme["center_core"], 255))

    def _draw_stamens(self, img, cx, cy):
        """绘制弯曲花丝和花药（从中心放射，无圆盘）"""
        theme = self.theme
        disc_r = self.big * 0.066
        filament_len = self.big * 0.032
        draw = ImageDraw.Draw(img)

        # 中心连接点（填补花丝汇聚处的空隙，带柔光）
        dot_r = self.big * 0.026
        dot_layer = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dot_layer)
        dd.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                   fill=(*theme["stamen"], 230))
        dot_layer = dot_layer.filter(ImageFilter.GaussianBlur(1.5))
        img.alpha_composite(dot_layer)

        for i in range(self.config["stamen_count"]):
            angle = (i / self.config["stamen_count"]) * math.tau + self.rng.uniform(-0.06, 0.06)
            sx = cx + math.cos(angle) * dot_r
            sy = cy + math.sin(angle) * dot_r
            length = filament_len * self.rng.uniform(0.75, 1.0)
            ex = cx + math.cos(angle) * (disc_r + length)
            ey = cy + math.sin(angle) * (disc_r + length)
            bend = self.rng.uniform(-0.15, 0.15)
            mx = (sx + ex) / 2 + -math.sin(angle) * length * bend
            my = (sy + ey) / 2 + math.cos(angle) * length * bend

            # 花丝（二次贝塞尔，分段画线）
            pts = []
            for t in np.linspace(0, 1, 12):
                pts.append(bezier_quad((sx, sy), (mx, my), (ex, ey), t))
            for j in range(len(pts) - 1):
                draw.line([pts[j], pts[j + 1]],
                          fill=(*theme["stamen"], 170),
                          width=max(1, int(self.big * 0.003)))

            # 花药
            ar = self.big * self.config["anther_size"]
            self._draw_anther(draw, (ex, ey), ar, theme["stamen_tip"])

    def _draw_anther(self, draw, pos, radius, color):
        """绘制单个花药，带花粉颗粒"""
        draw.ellipse([pos[0]-radius, pos[1]-radius,
                      pos[0]+radius, pos[1]+radius],
                     fill=(*color, 235))
        # 高光
        hr = radius * 0.35
        draw.ellipse([pos[0]-radius*0.3-hr, pos[1]-radius*0.3-hr,
                      pos[0]-radius*0.3+hr, pos[1]-radius*0.3+hr],
                     fill=(255, 245, 200, 120))

    def _draw_center_glow(self, img, cx, cy):
        """花蕊周围光晕"""
        glow = self._make_radial_gradient(
            self.big, cx, cy, self.big * 0.09,
            (255, 200, 80), (255, 200, 80), inner_alpha=22, outer_alpha=0
        )
        glow = glow.filter(ImageFilter.GaussianBlur(self.big * 0.02))
        img.alpha_composite(glow)

    def _make_radial_gradient(self, size, cx, cy, radius, inner_color, outer_color,
                              inner_alpha=255, outer_alpha=0):
        """使用numpy快速生成径向渐变"""
        x = np.arange(size, dtype=np.float64)
        y = np.arange(size, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        t = np.clip(d / radius, 0, 1)
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        for ch in range(3):
            arr[:, :, ch] = (inner_color[ch] + (outer_color[ch] - inner_color[ch]) * t).astype(np.uint8)
        arr[:, :, 3] = (inner_alpha + (outer_alpha - inner_alpha) * t).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")

    # ---------- 装饰 ----------
    def _draw_sparkles(self, img):
        """白色小圆点点缀（只在四角空白区域）"""
        layer = Image.new("RGBA", (self.big, self.big), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        cx, cy = self.big / 2, self.big / 2
        # 只在四个斜角方向放圆点，避开花瓣
        corners = [45, 135, 225, 315]
        for base_ang in corners:
            for _ in range(3):
                ang = math.radians(base_ang + self.rng.uniform(-12, 12))
                dist_v = self.rng.uniform(self.big * 0.42, self.big * 0.47)
                sx = cx + math.cos(ang) * dist_v
                sy = cy + math.sin(ang) * dist_v
                sr = self.rng.uniform(self.big * 0.002, self.big * 0.005)
                alpha = self.rng.randint(50, 120)
                d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr],
                          fill=(255, 255, 255, alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(self.big * 0.001))
        img.alpha_composite(layer)

    def _draw_texture(self, img):
        """极细微噪点纹理（numpy向量化，快速）"""
        intensity = self.config["texture_intensity"]
        if intensity <= 0:
            return
        count = int(self.big * self.big * 0.0008)
        # 用固定种子的numpy随机生成
        nrng = np.random.RandomState(42)
        xs = nrng.randint(0, self.big, count)
        ys = nrng.randint(0, self.big, count)
        vs = nrng.randint(-8, 9, count)
        alphas = nrng.randint(2, 7, count)
        arr = np.zeros((self.big, self.big, 4), dtype=np.uint8)
        vals = np.clip(128 + vs, 0, 255).astype(np.uint8)
        arr[ys, xs, 0] = vals
        arr[ys, xs, 1] = vals
        arr[ys, xs, 2] = vals
        arr[ys, xs, 3] = alphas
        noise = Image.fromarray(arr, "RGBA")
        img.alpha_composite(noise)

    def _draw_border(self, img):
        """精致边框"""
        draw = ImageDraw.Draw(img)
        radius = self.radius_big
        draw.rounded_rectangle([1, 1, self.big-2, self.big-2],
                               radius=radius-1, outline=(255, 255, 255, 30),
                               width=max(2, self.big // 450))
        draw.rounded_rectangle([3, 3, self.big-4, self.big-4],
                               radius=radius-3, outline=(0, 0, 0, 18), width=1)

    # ---------- 辅助 ----------
    def _make_rounded_mask(self, size, radius):
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, size-1, size-1],
                                               radius=radius, fill=255)
        return mask

# ══════════════════════════════════════════════════════
#  导出功能
# ══════════════════════════════════════════════════════

def apply_macos_mask(icon):
    """给图标应用 macOS 圆角矩形遮罩（squircle），四角变透明。"""
    from PIL import ImageDraw
    icon = icon.convert("RGBA")
    w, h = icon.size
    size = max(w, h)
    if w != h:
        s = min(w, h)
        icon = icon.crop(((w - s) // 2, (h - s) // 2,
                          (w + s) // 2, (h + s) // 2))
        size = s
    if size != 1024:
        icon = icon.resize((1024, 1024), Image.LANCZOS)
        size = 1024
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    radius = int(size * 0.2237)
    draw.rounded_rectangle([(0, 0), (size - 1, size - 1)], radius=radius, fill=255)
    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    result.paste(icon, (0, 0), mask)
    return result

def save_png(icon, path=PNG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    icon.save(path, "PNG")
    return path

def save_ico(icon, path=ICO_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sizes = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40),
             (48, 48), (64, 64), (128, 128), (256, 256)]
    icon.save(path, format="ICO", sizes=sizes)
    return path

def save_icns(icon, path=ICNS_PATH):
    import shutil
    import subprocess
    icon1024 = icon.convert("RGBA")
    if icon1024.size != (1024, 1024):
        icon1024 = icon1024.resize((1024, 1024), Image.LANCZOS)
    iconset = os.path.join(ASSETS_DIR, "icon.iconset")
    os.makedirs(iconset, exist_ok=True)
    sizes = {
        "icon_16x16.png": (16, 16),
        "icon_16x16@2x.png": (32, 32),
        "icon_32x32.png": (32, 32),
        "icon_32x32@2x.png": (64, 64),
        "icon_128x128.png": (128, 128),
        "icon_128x128@2x.png": (256, 256),
        "icon_256x256.png": (256, 256),
        "icon_256x256@2x.png": (512, 512),
        "icon_512x512.png": (512, 512),
        "icon_512x512@2x.png": (1024, 1024),
    }
    for name, sz in sizes.items():
        icon1024.resize(sz, Image.LANCZOS).save(os.path.join(iconset, name))
    try:
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", path],
                       check=True, capture_output=True, timeout=30)
        ok = True
    except Exception:
        ok = False
    shutil.rmtree(iconset, ignore_errors=True)
    return ok

def save_png_sizes(icon):
    for s in [16, 32, 64, 128, 256, 512]:
        p = os.path.join(ASSETS_DIR, f"icon_{s}.png")
        icon.resize((s, s), Image.LANCZOS).save(p)

def save_favicon(icon):
    path = os.path.join(ASSETS_DIR, "favicon.ico")
    icon.save(path, format="ICO", sizes=[(16, 16), (32, 32)])
    return path

def save_apple_touch(icon):
    path = os.path.join(ASSETS_DIR, "apple-touch-icon.png")
    icon.resize((180, 180), Image.LANCZOS).save(path)
    return path

def save_all_formats(icon, theme="midnight"):
    paths = {}
    # 不做圆角遮罩：macOS/Windows/Linux 会自动应用各自的图标遮罩，
    # 预先裁圆角会导致双重遮罩，出现白色边框。
    paths["png"] = save_png(icon)
    paths["ico"] = save_ico(icon)
    paths["favicon"] = save_favicon(icon)
    paths["apple"] = save_apple_touch(icon)
    if sys.platform == "darwin":
        if save_icns(icon):
            paths["icns"] = ICNS_PATH
    else:
        save_png_sizes(icon)
    for s in [64, 128, 256, 512]:
        p = os.path.join(ASSETS_DIR, f"icon_{s}.png")
        icon.resize((s, s), Image.LANCZOS).save(p)
        paths[f"png_{s}"] = p
    return paths

# ══════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════

def list_themes():
    print("可用配色主题：")
    for name in COLOR_THEMES:
        bg = COLOR_THEMES[name]["bg_mid"]
        petal = COLOR_THEMES[name]["petals"][0]
        print(f"  {name:10s}  背景RGB{bg}  花瓣RGB{petal}")

def main():
    parser = argparse.ArgumentParser(
        description="MrHua 精致樱花图标生成器（优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用主题:
  sakura  樱花紫（默认）
  night   夜色蓝紫
  dawn    晨曦粉橙
  ocean   海洋蓝
  forest  森林绿

示例:
  python generate_icon.py
  python generate_icon.py --theme night
  python generate_icon.py --all-themes
  python generate_icon.py --size 512
        """)
    parser.add_argument("--theme", default="midnight",
                        choices=list(COLOR_THEMES.keys()))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--all-themes", action="store_true")
    parser.add_argument("--list-themes", action="store_true")
    args = parser.parse_args()

    if args.list_themes:
        list_themes()
        return

    os.makedirs(ASSETS_DIR, exist_ok=True)
    try:
        from PIL import Image  # noqa
        import numpy  # noqa
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("安装: pip install pillow numpy")
        sys.exit(1)

    if args.all_themes:
        for theme_name in COLOR_THEMES:
            print(f"🌸 生成主题: {theme_name}")
            renderer = IconRenderer(args.size, theme_name)
            icon = renderer.render()
            theme_png = os.path.join(ASSETS_DIR, f"icon_{theme_name}.png")
            icon.save(theme_png, "PNG")
            print(f"  ✅ {theme_png}")
        print("\n完成。")
        return

    print(f"🌸 绘制 MrHua 樱花图标（主题: {args.theme}, {args.size}px, {DEFAULT_CONFIG['ssaa']}x超采样）...")
    renderer = IconRenderer(args.size, args.theme)
    icon = renderer.render()

    # 不做圆角遮罩：macOS/Windows/Linux 会自动应用各自的图标遮罩，
    # 预先裁圆角会导致双重遮罩，出现白色边框。

    print("  保存 PNG...")
    save_png(icon)
    print(f"  ✅ {PNG_PATH}")

    print("  保存 ICO（多尺寸）...")
    save_ico(icon)
    print(f"  ✅ {ICO_PATH}")

    print("  保存 favicon...")
    save_favicon(icon)
    print(f"  ✅ {os.path.join(ASSETS_DIR, 'favicon.ico')}")

    print("  保存 Apple Touch Icon...")
    save_apple_touch(icon)
    print(f"  ✅ {os.path.join(ASSETS_DIR, 'apple-touch-icon.png')}")

    if sys.platform == "darwin":
        print("  生成 ICNS...")
        if save_icns(icon):
            print(f"  ✅ {ICNS_PATH}")
        else:
            print("  ⚠️ ICNS 生成失败")
    else:
        save_png_sizes(icon)
        print(f"  ✅ 多尺寸 PNG 已生成到 {ASSETS_DIR}")

    print("\n🎉 图标生成完成。")

if __name__ == "__main__":
    main()