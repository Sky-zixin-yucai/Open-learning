# 03_旋转矩形检测.py
"""

旋转矩形计算和绘制
Rotating rectangle calculation and drawing

"""
import cv2
import numpy as np
import math

class RotatedRectDetector:
    """旋转矩形检测类"""
    
    def __init__(self):
        self.rotated_rects = []
        
    def detect_rotated_rectangles(self, contours, min_area=50, min_aspect_ratio=0.1, max_aspect_ratio=10):
        """
        从轮廓检测旋转矩形
        
        参数:
            contours: 轮廓列表
            min_area: 最小面积阈值
            min_aspect_ratio: 最小宽高比
            max_aspect_ratio: 最大宽高比
            
        返回:
            旋转矩形列表，每个元素为(center, size, angle)
        """
        rotated_rects = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # 计算最小外接旋转矩形
            rect = cv2.minAreaRect(contour)
            
            # 获取矩形参数
            center, size, angle = rect
            width, height = size
            
            # 调整角度：OpenCV返回的角度范围是[-90, 0)
            # 我们将其转换为[0, 180)
            if width < height:
                angle = angle + 90
                width, height = height, width
                
            # 过滤太窄或太宽的矩形
            if height == 0:
                continue
                
            aspect_ratio = width / height
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
                
            # 存储旋转矩形
            rotated_rects.append({
                'center': center,
                'size': (width, height),
                'angle': angle,
                'area': area,
                'contour': contour
            })
            
        self.rotated_rects = rotated_rects
        return rotated_rects
    
    def get_rotated_box_points(self, rotated_rect):
        """
        获取旋转矩形的四个角点
        
        参数:
            rotated_rect: 旋转矩形字典
            
        返回:
            四个角点的坐标，形状为(4, 2)
        """
        center = rotated_rect['center']
        size = rotated_rect['size']
        angle = rotated_rect['angle']
        
        # 创建旋转矩形对象
        rect = (center, size, angle)
        
        # 获取四个角点
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        return box
    
    def draw_rotated_rectangles(self, image, rotated_rects, draw_center=True, draw_angle=True):
        """
        绘制旋转矩形
        
        参数:
            image: 输入图像
            rotated_rects: 旋转矩形列表
            draw_center: 是否绘制中心点
            draw_angle: 是否显示角度
            
        返回:
            绘制了旋转矩形的图像
        """
        result = image.copy()
        
        for i, rect in enumerate(rotated_rects):
            # 获取旋转矩形的颜色（使用彩虹色）
            color = self.get_color_by_index(i)
            
            # 获取四个角点
            box = self.get_rotated_box_points(rect)
            
            # 绘制旋转矩形边框
            cv2.drawContours(result, [box], 0, color, 2)
            
            # 绘制中心点
            if draw_center:
                center = tuple(map(int, rect['center']))
                cv2.circle(result, center, 5, color, -1)
                
            # 显示角度
            if draw_angle:
                angle_text = f"{rect['angle']:.1f}°"
                center = tuple(map(int, rect['center']))
                text_pos = (center[0] + 10, center[1] - 10)
                cv2.putText(result, angle_text, text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 显示面积
            area_text = f"A:{rect['area']:.0f}"
            text_pos = (center[0] + 10, center[1] + 20)
            cv2.putText(result, area_text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return result
    
    def get_color_by_index(self, index):
        """根据索引获取彩虹色"""
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (0, 128, 128),  # 茶色
        ]
        return colors[index % len(colors)]
    
    def calculate_iou(self, rect1, rect2):
        """
        计算两个旋转矩形的交并比
        
        参数:
            rect1: 第一个旋转矩形
            rect2: 第二个旋转矩形
            
        返回:
            交并比(IoU)
        """
        # 获取两个矩形的角点
        box1 = self.get_rotated_box_points(rect1)
        box2 = self.get_rotated_box_points(rect2)
        
        # 计算相交区域
        intersection = self.polygon_intersection_area(box1, box2)
        
        # 计算各自的面积
        area1 = rect1['area']
        area2 = rect2['area']
        
        # 计算并集面积
        union = area1 + area2 - intersection
        
        # 计算IoU
        if union == 0:
            return 0
            
        iou = intersection / union
        return iou
    
    def polygon_intersection_area(self, poly1, poly2):
        """
        计算两个凸多边形的相交面积
        简化版本：使用旋转矩形的近似计算
        """
        # 将多边形转换为轮廓
        poly1_contour = poly1.reshape(-1, 1, 2).astype(np.float32)
        poly2_contour = poly2.reshape(-1, 1, 2).astype(np.float32)
        
        # 计算相交多边形
        try:
            _, intersection = cv2.intersectConvexConvex(poly1_contour, poly2_contour)
            if len(intersection) > 0:
                return cv2.contourArea(intersection)
        except:
            pass
            
        return 0
    
    def filter_by_iou(self, rotated_rects, iou_threshold=0.5):
        """
        根据IoU过滤重叠的旋转矩形
        
        参数:
            rotated_rects: 旋转矩形列表
            iou_threshold: IoU阈值，大于此值视为重叠
            
        返回:
            过滤后的旋转矩形列表
        """
        if not rotated_rects:
            return []
            
        # 按面积降序排序
        sorted_rects = sorted(rotated_rects, key=lambda x: x['area'], reverse=True)
        
        filtered = []
        
        for i, rect in enumerate(sorted_rects):
            # 检查是否与已选中的矩形重叠
            keep = True
            
            for kept_rect in filtered:
                iou = self.calculate_iou(rect, kept_rect)
                if iou > iou_threshold:
                    keep = False
                    break
                    
            if keep:
                filtered.append(rect)
                
        return filtered

# 使用示例
if __name__ == "__main__":
    from ddoo1 import ImageProcessor
    from ddoo11 import EdgeDetector
    
    # 创建处理器
    processor = ImageProcessor()
    edge_detector = EdgeDetector()
    rect_detector = RotatedRectDetector()
    
    # 读取图像
    image_path = r'D:\桌面\学习\python旋转目标检测\dota_demo.png'
    if processor.read_image(image_path):
        # 转换为灰度图
        gray = processor.to_gray()
        
        # 边缘检测
        edges = edge_detector.detect_edges(gray, 'canny', low_threshold=30, high_threshold=100)
        
        # 查找轮廓
        contours = edge_detector.find_contours(edges, mode='external')
        filtered_contours = edge_detector.filter_contours(contours, min_area=100)
        
        # 检测旋转矩形
        rotated_rects = rect_detector.detect_rotated_rectangles(filtered_contours, min_area=100)
        print(f"检测到 {len(rotated_rects)} 个旋转矩形")
        
        # 过滤重叠矩形
        filtered_rects = rect_detector.filter_by_iou(rotated_rects, iou_threshold=0.3)
        print(f"过滤后剩余 {len(filtered_rects)} 个旋转矩形")
        
        # 绘制旋转矩形
        result_image = rect_detector.draw_rotated_rectangles(
            processor.image, filtered_rects, draw_center=True, draw_angle=True
        )
        
        # 显示结果
        processor.show_image("旋转矩形检测", result_image, wait_key=0)
        
        # 保存结果
        processor.save_image("rotated_rect_detection.jpg", result_image)