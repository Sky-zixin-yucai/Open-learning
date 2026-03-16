# 01_基础图像处理.py
"""

图像读写和基本操作
Image reading, writing, and basic operations

"""
import cv2
import numpy as np
import os

class ImageProcessor:
    """图像处理基础类"""
    
    def __init__(self):
        self.image = None
        self.height = 0
        self.width = 0
        
    def read_image(self, image_path):
        """
        读取图像（支持中文路径）
        
        参数:
            image_path: 图像文件路径
            
        返回:
            是否读取成功
        """
        # 方法1: 使用imdecode读取（支持中文路径）
        try:
            with open(image_path, 'rb') as f:
                img_bytes = np.frombuffer(f.read(), np.uint8)
                self.image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"读取图像失败: {e}")
            return False
        
        if self.image is None:
            print("图像文件可能损坏或格式不支持")
            return False
            
        self.height, self.width = self.image.shape[:2]
        print(f"图像读取成功！尺寸: {self.width}x{self.height}, 通道: {self.image.shape[2]}")
        return True
    
    def save_image(self, output_path, image=None):
        """
        保存图像（支持中文路径）
        
        参数:
            output_path: 输出路径
            image: 要保存的图像，如果为None则保存self.image
        """
        if image is None:
            image = self.image
            
        if image is None:
            print("没有图像可保存")
            return False
            
        try:
            # 获取文件扩展名
            ext = os.path.splitext(output_path)[1]
            # 编码图像
            success, encoded = cv2.imencode(ext, image)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded.tobytes())
                print(f"图像保存成功: {output_path}")
                return True
            return False
        except Exception as e:
            print(f"保存图像失败: {e}")
            return False
    
    def show_image(self, window_name="Image", image=None, wait_key=0):
        """
        显示图像
        
        参数:
            window_name: 窗口名称
            image: 要显示的图像
            wait_key: 等待时间(0=无限等待)
        """
        if image is None:
            image = self.image
            
        if image is None:
            print("没有图像可显示")
            return
            
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_key)
        cv2.destroyAllWindows()
    
    def to_gray(self, image=None):
        """
        转换为灰度图像
        
        参数:
            image: 输入图像
            
        返回:
            灰度图像
        """
        if image is None:
            image = self.image
            
        if image is None:
            print("没有图像可转换")
            return None
            
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        return gray
    
    def resize_image(self, new_width=None, new_height=None, scale_factor=None, image=None):
        """
        调整图像大小
        
        参数:
            new_width: 新宽度
            new_height: 新高度
            scale_factor: 缩放比例
            image: 输入图像
            
        返回:
            调整大小后的图像
        """
        if image is None:
            image = self.image
            
        if image is None:
            print("没有图像可调整")
            return None
            
        height, width = image.shape[:2]
        
        if scale_factor is not None:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        elif new_width is None and new_height is not None:
            scale = new_height / height
            new_width = int(width * scale)
        elif new_height is None and new_width is not None:
            scale = new_width / width
            new_height = int(height * scale)
        elif new_width is None and new_height is None:
            print("请指定宽度、高度或缩放比例")
            return None
            
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized

# 使用示例
if __name__ == "__main__":
    # 创建图像处理器
    processor = ImageProcessor()
    
    # 读取图像
    image_path = r'D:\桌面\学习\python旋转目标检测\dota_demo.png'
    if processor.read_image(image_path):
        # 显示原图
        processor.show_image("原图", wait_key=1000)
        
        # 转换为灰度图
        gray = processor.to_gray()
        processor.show_image("灰度图", gray, wait_key=1000)
        
        # 调整大小
        resized = processor.resize_image(new_width=800)
        processor.show_image("调整大小", resized, wait_key=1000)
        
        # 保存图像
        processor.save_image("test_output.jpg", resized)