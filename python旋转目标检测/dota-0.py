import cv2
import numpy as np
from typing import List, Tuple

def create_rotated_gaussian(height: int, width: int, 
                           center: Tuple[int, int], 
                           sigma: Tuple[float, float], 
                           theta_deg: float) -> np.ndarray:
    """
    创建旋转的椭圆形2D高斯分布
    
    参数:
        height: 图像高度
        width: 图像宽度
        center: 高斯中心坐标 (x, y)
        sigma: 高斯标准差 (sigma_x, sigma_y)
        theta_deg: 旋转角度（度数）
        
    返回:
        高斯分布图像
    """
    center_x, center_y = center
    sigma_x, sigma_y = sigma
    theta = np.radians(theta_deg)
    
    # 创建坐标网格
    x = np.arange(width) - center_x
    y = np.arange(height) - center_y
    X, Y = np.meshgrid(x, y)
    
    # 旋转坐标系
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    X_rot = X * cos_theta + Y * sin_theta
    Y_rot = -X * sin_theta + Y * cos_theta
    
    # 计算旋转后椭圆形高斯分布的值
    gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
    
    return gaussian

def main():
    # 读取输入图像
    input_image_path = 'dota_demo.png'  # 请替换为您的图像路径
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {input_image_path}")
    
    # 获取图像尺寸
    height, width = image.shape
    
    # 高斯参数列表: [(center_x, center_y), (sigma_x, sigma_y), theta_deg]
    gaussian_params = [
        ((272, 462), (80, 20), 25),   # 高斯1
        ((799, 359), (20, 20), 15),   # 高斯2
        ((709, 502), (10, 20), -20),  # 高斯3
        ((746, 371), (100, 20), -80), # 高斯4
    ]
    
    # 创建一个与输入图像相同尺寸的空白图像用于绘制高斯
    gaussian_image = np.zeros((height, width), dtype=np.float32)
    
    # 为每个高斯参数生成并叠加高斯分布
    for center, sigma, theta_deg in gaussian_params:
        gaussian = create_rotated_gaussian(height, width, center, sigma, theta_deg)
        gaussian_image += gaussian
    
    # 将高斯图像归一化到0-255
    gaussian_image = cv2.normalize(gaussian_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # 转换为8位图像
    gaussian_image = gaussian_image.astype(np.uint8)
    
    # 应用颜色映射
    colored_gaussian = cv2.applyColorMap(gaussian_image, cv2.COLORMAP_JET)
    
    # 将输入图像转换为三通道并叠加
    input_image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    output_image = cv2.addWeighted(input_image_color, 0.7, colored_gaussian, 0.3, 0)
    
    # 保存结果图像
    output_image_path = 'output_example_gs.png'
    cv2.imwrite(output_image_path, output_image)
    
    # 可选：显示结果
    # cv2.imshow('Output', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()