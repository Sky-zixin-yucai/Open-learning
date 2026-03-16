# 02_边缘轮廓检测.py
"""

边缘检测和轮廓提取
Edge detection and contour extraction

"""
import cv2
import numpy as np


class EdgeDetector:
    """边缘和轮廓检测类"""
    
    def __init__(self):
        self.contours = []
        self.hierarchy = None
        
    def detect_edges(self, image, method='canny', **kwargs):
        """
        边缘检测
        
        参数:
            image: 输入图像（灰度图）
            method: 边缘检测方法，可选'canny'或'sobel'
            **kwargs: 方法参数
            
        返回:
            边缘图像
        """
        if method == 'canny':
            # 默认参数
            low_threshold = kwargs.get('low_threshold', 50)
            high_threshold = kwargs.get('high_threshold', 150)
            edges = cv2.Canny(image, low_threshold, high_threshold)
            
        elif method == 'sobel':
            # Sobel算子边缘检测
            scale = kwargs.get('scale', 1)
            delta = kwargs.get('delta', 0)
            ddepth = kwargs.get('ddepth', cv2.CV_16S)
            
            # X方向梯度
            grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta)
            grad_x = cv2.convertScaleAbs(grad_x)
            
            # Y方向梯度
            grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta)
            grad_y = cv2.convertScaleAbs(grad_y)
            
            # 合并梯度
            edges = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        else:
            raise ValueError(f"不支持的边缘检测方法: {method}")
            
        return edges
    
    def find_contours(self, edge_image, mode='external', method='simple'):
        """
        查找轮廓
        
        参数:
            edge_image: 边缘图像
            mode: 轮廓检索模式
                'external': 只检测外轮廓
                'list': 所有轮廓，不建立层次关系
                'tree': 建立轮廓的完整层次结构
            method: 轮廓近似方法
                'simple': 压缩水平、垂直和对角线段，只保留端点
                'none': 存储所有轮廓点
                
        返回:
            轮廓列表
        """
        # 转换为二值图像
        if len(edge_image.shape) > 2:
            edge_image = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
            
        _, binary = cv2.threshold(edge_image, 1, 255, cv2.THRESH_BINARY)
        
        # 设置轮廓检索模式
        if mode == 'external':
            mode_flag = cv2.RETR_EXTERNAL
        elif mode == 'list':
            mode_flag = cv2.RETR_LIST
        elif mode == 'tree':
            mode_flag = cv2.RETR_TREE
        else:
            mode_flag = cv2.RETR_EXTERNAL
            
        # 设置轮廓近似方法
        if method == 'simple':
            method_flag = cv2.CHAIN_APPROX_SIMPLE
        elif method == 'none':
            method_flag = cv2.CHAIN_APPROX_NONE
        else:
            method_flag = cv2.CHAIN_APPROX_SIMPLE
            
        # 查找轮廓
        contours, hierarchy = cv2.findContours(binary, mode_flag, method_flag)
        
        self.contours = contours
        self.hierarchy = hierarchy
        
        return contours
    
    def filter_contours(self, contours, min_area=100, max_area=None, aspect_ratio_range=(0.1, 10)):
        """
        过滤轮廓
        
        参数:
            contours: 轮廓列表
            min_area: 最小面积阈值
            max_area: 最大面积阈值
            aspect_ratio_range: 宽高比范围
            
        返回:
            过滤后的轮廓列表
        """
        filtered = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < min_area:
                continue
                
            if max_area is not None and area > max_area:
                continue
            
            # 计算轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 避免除零错误
            if h == 0:
                continue
                
            # 宽高比过滤
            aspect_ratio = w / h
            min_ratio, max_ratio = aspect_ratio_range
            
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                continue
                
            filtered.append(contour)
            
        return filtered
    
    def draw_contours(self, image, contours, color=(0, 255, 0), thickness=2):
        """
        绘制轮廓
        
        参数:
            image: 输入图像
            contours: 轮廓列表
            color: 颜色(B,G,R)
            thickness: 线宽
            
        返回:
            绘制了轮廓的图像
        """
        result = image.copy()
        
        # 绘制所有轮廓
        cv2.drawContours(result, contours, -1, color, thickness)
        
        return result
    
    def get_contour_info(self, contour):
        """
        获取轮廓信息
        
        参数:
            contour: 轮廓
            
        返回:
            轮廓信息字典
        """
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算轮廓中心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
            
        # 计算圆形度
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
            
        return {
            'area': area,
            'perimeter': perimeter,
            'bounding_box': (x, y, w, h),
            'center': (cx, cy),
            'circularity': circularity
        }

# 使用示例
if __name__ == "__main__":
    from ddoo1 import ImageProcessor
    
    # 创建图像处理器
    processor = ImageProcessor()
    detector = EdgeDetector()
    
    # 读取图像
    image_path = r'D:\桌面\学习\python旋转目标检测\dota_demo.png'
    if processor.read_image(image_path):
        # 转换为灰度图
        gray = processor.to_gray()
        
        # 边缘检测
        edges = detector.detect_edges(gray, 'canny', low_threshold=30, high_threshold=100)
        
        # 显示边缘
        processor.show_image("边缘检测", edges, wait_key=1000)
        
        # 查找轮廓
        contours = detector.find_contours(edges, mode='external')
        print(f"找到 {len(contours)} 个轮廓")
        
        # 过滤轮廓
        filtered_contours = detector.filter_contours(contours, min_area=100, max_area=10000)
        print(f"过滤后剩余 {len(filtered_contours)} 个轮廓")
        
        # 绘制轮廓
        contour_image = detector.draw_contours(processor.image, filtered_contours)
        processor.show_image("轮廓检测", contour_image, wait_key=1000)
        
        # 获取轮廓信息
        for i, contour in enumerate(filtered_contours[:5]):  # 只显示前5个
            info = detector.get_contour_info(contour)
            print(f"轮廓{i+1}: 面积={info['area']:.1f}, 中心={info['center']}")