import cv2
import numpy as np
import os

def main():
    # 使用绝对路径（根据实际情况修改）
    input_image_path = r'D:\桌面\学习\python旋转目标检测\dota_demo.png'
    
    # 检查文件是否存在
    if not os.path.isfile(input_image_path):
        print(f"文件不存在: {input_image_path}")
        print("请确认文件路径是否正确")
        return
    
    print(f"正在读取图像: {input_image_path}")
    
    # 读取图像（带错误处理）
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # 如果读取失败，尝试其他方法
    if image is None:
        print("标准读取失败，尝试其他方法...")
        
        # 方法1: 尝试读取为彩色然后转换
        image_bgr = cv2.imread(input_image_path)
        if image_bgr is not None:
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            print("成功以彩色模式读取并转换为灰度")
        else:
            # 方法2: 使用imdecode读取
            try:
                with open(input_image_path, 'rb') as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    print("使用imdecode读取成功")
            except Exception as e:
                print(f"所有读取方法均失败: {e}")
                return
    
    if image is None:
        print("无法读取图像，可能是文件损坏或不支持的格式")
        return
    
    # 获取图像尺寸
    height, width = image.shape
    
    print(f"图像读取成功！尺寸: {width}x{height}")
    
    # 继续原来的高斯生成代码...
    # 设置高斯分布的参数
    gaussian_params = [
        ((272, 462), (80, 20), 25),
        ((799, 359), (20, 20), 15),
        ((709, 502), (10, 20), -20),
        ((746, 371), (100, 20), -80),
    ]
    
    # 创建一个与输入图像相同尺寸的空白图像用于绘制高斯
    gaussian_image = np.zeros((height, width), dtype=np.float32)
    
    # 生成高斯分布
    for center, sigma, theta_deg in gaussian_params:
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
    output_image_path = r'D:\桌面\学习\python旋转目标检测\output_example_gs.png'
    cv2.imwrite(output_image_path, output_image)
    
    print(f"处理完成，结果保存到: {output_image_path}")
    
    # 可选：显示结果
    # cv2.imshow('Input', image)
    # cv2.imshow('Output', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()