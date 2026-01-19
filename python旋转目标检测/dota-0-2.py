import cv2
import numpy as np
import os

def cv2_read_image_chinese(image_path, flags=cv2.IMREAD_GRAYSCALE):
    """
    支持中文路径的OpenCV图像读取函数
    
    参数:
        image_path: 图像路径（可以是中文路径）
        flags: OpenCV读取标志
        
    返回:
        读取的图像，如果失败返回None
    """
    # 尝试使用imdecode读取
    try:
        with open(image_path, 'rb') as f:
            img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, flags)
        return img
    except Exception as e:
        print(f"使用imdecode读取失败: {e}")
        return None

def main():
    # 使用绝对路径
    input_image_path = r'D:\桌面\学习\python旋转目标检测\dota_demo.png'
    output_image_path = r'D:\桌面\学习\python旋转目标检测\output_example_gs.png'
    
    # 检查文件是否存在
    if not os.path.isfile(input_image_path):
        print(f"错误：文件不存在: {input_image_path}")
        return
    
    print(f"正在读取图像: {input_image_path}")
    
    # 使用支持中文路径的读取函数
    image = cv2_read_image_chinese(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        # 如果灰度读取失败，尝试读取彩色然后转换
        image_color = cv2_read_image_chinese(input_image_path, cv2.IMREAD_COLOR)
        if image_color is not None:
            image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
            print("成功以彩色模式读取并转换为灰度")
    
    if image is None:
        print("无法读取图像，请检查文件是否损坏")
        return
    
    # 获取图像尺寸
    height, width = image.shape
    print(f"图像读取成功！尺寸: {width}x{height}")
    
    # 高斯参数配置
    gaussian_params = [
        ((272, 462), (80, 20), 25),   # 高斯1
        ((799, 359), (20, 20), 15),   # 高斯2
        ((709, 502), (10, 20), -20),  # 高斯3
        ((746, 371), (100, 20), -80), # 高斯4
    ]
    
    # 创建高斯图像
    gaussian_image = np.zeros((height, width), dtype=np.float32)
    
    # 生成所有高斯分布
    for center, sigma, theta_deg in gaussian_params:
        center_x, center_y = center
        sigma_x, sigma_y = sigma
        theta = np.radians(theta_deg)
        
        # 创建坐标网格（向量化操作，性能更好）
        x = np.arange(width) - center_x
        y = np.arange(height) - center_y
        X, Y = np.meshgrid(x, y)
        
        # 旋转坐标系
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        X_rot = X * cos_theta + Y * sin_theta
        Y_rot = -X * sin_theta + Y * cos_theta
        
        # 计算旋转后椭圆形高斯分布的值
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
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
    
    # 保存结果图像（也需要支持中文路径）
    def cv2_write_image_chinese(image_path, image):
        """支持中文路径的OpenCV图像保存函数"""
        try:
            # 获取文件扩展名
            ext = os.path.splitext(image_path)[1]
            # 使用imencode编码图像
            success, encoded_image = cv2.imencode(ext, image)
            if success:
                with open(image_path, 'wb') as f:
                    f.write(encoded_image.tobytes())
                return True
            return False
        except Exception as e:
            print(f"保存图像失败: {e}")
            return False
    
    # 保存图像
    if cv2_write_image_chinese(output_image_path, output_image):
        print(f"处理完成，结果保存到: {output_image_path}")
    else:
        # 如果中文路径保存失败，尝试使用英文路径
        safe_output_path = 'output_example_gs.png'
        cv2.imwrite(safe_output_path, output_image)
        print(f"使用中文路径保存失败，结果已保存到: {safe_output_path}")

if __name__ == "__main__":
    main()