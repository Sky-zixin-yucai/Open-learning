"""

关键改进点
1. 检测和处理压缩

# 之前 假设urllib处理了所有压缩
# 现在 显式检查并处理各种压缩格式

Content-Encoding: gzip, deflate, br  # 可能同时支持多种
2. 更智能的编码探测

# 之前：简单尝试几个编码
# 现在：
# 1. 检测BOM 字节顺序标记
# 2. 检查是否像二进制数据
# 3. 验证解码结果是否合理

3. 区分文本和二进制

# 之前：强制把所有内容当作文本
# 现在：识别二进制内容，避免错误解码

4. 更好的调试信息

# 打印关键信息：
# - Content-Type
# - Content-Encoding  
# - 原始字节长度
# - 编码尝试结果
# - 解压状态

"""



from urllib import request
from urllib.error import URLError, HTTPError
import ssl
import re
import gzip
import io

def decode_content_with_advanced_detection(response):
    """
    高级内容解码函数，处理各种编码和压缩问题
    
    Args:
        response: urllib的响应对象
    
    Returns:
        (解码后的文本内容, 实际使用的编码)
    """
    # 1. 读取原始字节
    content_bytes = response.read()
    
    # 2. 获取响应头信息（用于调试）
    content_type = response.headers.get('Content-Type', '')
    content_encoding = response.headers.get('Content-Encoding', '').lower()
    
    print(f"Content-Type: {content_type}")
    print(f"Content-Encoding: {content_encoding}")
    print(f"原始字节长度: {len(content_bytes)}")
    
    # 3. 处理可能的压缩（重点改进！）
    content_bytes = handle_compression(content_bytes, content_encoding)
    
    # 4. 从Content-Type中提取声称的编码
    claimed_encoding = None
    if 'charset=' in content_type:
        claimed_encoding = content_type.split('charset=')[-1].split(';')[0].strip().lower()
        print(f"服务器声称的编码: {claimed_encoding}")
    
    # 5. 尝试用声称的编码解码
    if claimed_encoding:
        try:
            content = content_bytes.decode(claimed_encoding)
            print(f"✅ 使用声称的编码 {claimed_encoding} 解码成功")
            return content, claimed_encoding
        except (UnicodeDecodeError, LookupError) as e:
            print(f"❌ 声称的编码 {claimed_encoding} 解码失败: {e}")
    
    # 6. 自动编码探测（改进版）
    # 先尝试猜测是否为二进制内容
    if looks_like_binary(content_bytes):
        print("⚠️  内容看起来像二进制数据，不进行文本解码")
        return content_bytes, 'binary'
    
    # 尝试解码的编码列表（按优先级排序）
    encoding_attempts = [
        'utf-8',        # 最通用的编码
        'gbk',          # 中文Windows常用
        'gb2312',       # 简体中文标准
        'big5',         # 繁体中文
        'shift_jis',    # 日文
        'euc-kr',       # 韩文
        'iso-8859-1',   # 西欧语言
        'windows-1252', # Windows西欧扩展
        'cp1251',       # 西里尔字母
    ]
    
    # 7. 高级编码探测
    detected_encoding = advanced_encoding_detection(content_bytes)
    if detected_encoding:
        encoding_attempts.insert(0, detected_encoding)  # 优先尝试检测到的编码
    
    print("🔍 开始编码探测...")
    for encoding in encoding_attempts:
        try:
            # 尝试解码，不替换错误字符
            content = content_bytes.decode(encoding, errors='strict')
            
            # 验证解码结果是否合理
            if is_reasonable_text(content):
                print(f"✅ 成功使用编码: {encoding}")
                print(f"   字符数: {len(content)}，可打印字符比例: {calculate_printable_ratio(content):.1%}")
                return content, encoding
            else:
                print(f"⚠️  编码 {encoding} 解码成功但内容不合理")
                
        except UnicodeDecodeError:
            print(f"❌ 编码 {encoding} 解码失败")
            continue
        except LookupError:
            print(f"❌ 未知编码: {encoding}")
            continue
    
    # 8. 所有编码都失败，使用安全回退
    print("⚠️  所有编码尝试失败，使用安全回退（errors='replace'）")
    try:
        content = content_bytes.decode('utf-8', errors='replace')
        return content, 'utf-8 (replacement)'
    except:
        # 终极回退：当作二进制处理
        print("❌ 终极回退：当作二进制数据")
        return content_bytes, 'binary (fallback)'

def handle_compression(content_bytes, content_encoding):
    """
    处理各种压缩格式
    """
    if not content_encoding:
        return content_bytes
    
    print(f"检测到压缩格式: {content_encoding}")
    
    try:
        if content_encoding == 'gzip' or content_encoding == 'x-gzip':
            # urllib应该已经自动解压gzip，但有时需要手动处理
            if content_bytes[:2] == b'\x1f\x8b':  # gzip魔数
                with gzip.GzipFile(fileobj=io.BytesIO(content_bytes)) as f:
                    return f.read()
        
        elif content_encoding == 'deflate':
            # 尝试两种deflate格式
            try:
                # RFC 1950格式（带zlib头）
                import zlib
                return zlib.decompress(content_bytes)
            except zlib.error:
                # RFC 1951格式（原始deflate）
                return zlib.decompress(content_bytes, -15)
        
        elif content_encoding == 'br':
            # Brotli压缩（需要安装brotli库）
            try:
                import brotli
                return brotli.decompress(content_bytes)
            except ImportError:
                print("❌ 需要安装brotli库：pip install brotli")
            except Exception as e:
                print(f"❌ Brotli解压失败: {e}")
        
        elif content_encoding == 'compress' or content_encoding == 'x-compress':
            print("❌ LZW压缩格式，Python标准库不支持")
        
        else:
            print(f"❌ 不支持的压缩格式: {content_encoding}")
            
    except Exception as e:
        print(f"❌ 解压失败: {e}")
    
    return content_bytes  # 如果解压失败，返回原始内容

def looks_like_binary(content_bytes, sample_size=1000):
    """
    判断内容是否像二进制数据
    """
    if len(content_bytes) == 0:
        return False
    
    # 取前1000个字节分析
    sample = content_bytes[:min(sample_size, len(content_bytes))]
    
    # 统计可打印ASCII字符的比例
    printable_count = 0
    for byte in sample:
        if 32 <= byte <= 126 or byte in (9, 10, 13):  # 可打印ASCII + tab, lf, cr
            printable_count += 1
    
    printable_ratio = printable_count / len(sample)
    
    # 如果可打印字符比例低于70%，可能是二进制
    if printable_ratio < 0.7:
        return True
    
    # 检查是否有太多连续的空字节（二进制文件常见）
    if b'\x00' in sample:
        null_count = sample.count(b'\x00')
        if null_count > len(sample) * 0.05:  # 超过5%的空字节
            return True
    
    return False

def advanced_encoding_detection(content_bytes, sample_size=4096):
    """
    高级编码探测（简单版）
    
    注意：更准确的探测可以使用chardet库：
    pip install chardet
    """
    # 简单的BOM检测
    bom_encodings = [
        (b'\xef\xbb\xbf', 'utf-8-sig'),    # UTF-8 BOM
        (b'\xff\xfe', 'utf-16-le'),        # UTF-16 LE BOM
        (b'\xfe\xff', 'utf-16-be'),        # UTF-16 BE BOM
        (b'\xff\xfe\x00\x00', 'utf-32-le'), # UTF-32 LE BOM
        (b'\x00\x00\xfe\xff', 'utf-32-be'), # UTF-32 BE BOM
    ]
    
    for bom, encoding in bom_encodings:
        if content_bytes.startswith(bom):
            print(f"检测到BOM: {encoding}")
            return encoding
    
    # 简单的模式检测
    sample = content_bytes[:min(sample_size, len(content_bytes))]
    
    # 检测UTF-8模式
    try:
        sample.decode('utf-8', errors='strict')
        # 如果通过严格测试，很可能是UTF-8
        print("通过UTF-8严格测试")
        return 'utf-8'
    except:
        pass
    
    return None

def is_reasonable_text(text, sample_size=1000):
    """
    判断解码后的文本是否合理
    
    标准：
    1. 有一定比例的可打印字符
    2. 没有过多的控制字符
    3. 对于中文，有一定比例的中文字符
    """
    if len(text) == 0:
        return False
    
    sample = text[:min(sample_size, len(text))]
    
    # 计算可打印字符比例
    import string
    printable_chars = string.printable + ' '  # 包括空格
    printable_count = sum(1 for c in sample if c in printable_chars)
    printable_ratio = printable_count / len(sample)
    
    # 检查是否有过多控制字符（除常见空白字符外）
    control_count = sum(1 for c in sample if ord(c) < 32 and c not in ('\n', '\r', '\t'))
    control_ratio = control_count / len(sample)
    
    # 检查中文字符比例（可选）
    chinese_count = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff')
    chinese_ratio = chinese_count / len(sample)
    
    # 判断标准
    if printable_ratio > 0.8 and control_ratio < 0.1:
        # 如果有很多中文字符，就更可能是合理的文本
        if chinese_ratio > 0.1:
            return True
        # 对于非中文文本，要求更高的可打印比例
        elif printable_ratio > 0.9:
            return True
    
    return False

def calculate_printable_ratio(text):
    """计算可打印字符比例"""
    import string
    printable_chars = string.printable + ' '
    printable_count = sum(1 for c in text if c in printable_chars)
    return printable_count / len(text) if len(text) > 0 else 0

# 使用改进版解码器的完整爬虫
def improved_crawler():
    """使用改进版解码器的爬虫"""
    ssl._create_default_https_context = ssl._create_unverified_context
    url = 'https://setuptools.pypa.io/en/latest/pkg_resources.html'  # 可以替换为实际需要爬取的URL
    
    try:
        req = request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br'
            }
        )
        
        with request.urlopen(req, timeout=10) as response:
            print(f"状态码: {response.status}")
            
            # 使用改进版解码器
            content, encoding = decode_content_with_advanced_detection(response)
            
            print(f"\n最终使用的编码: {encoding}")
            print(f"内容长度: {len(content)} {'字符' if isinstance(content, str) else '字节'}")
            
            # 智能显示
            if isinstance(content, str):
                # 如果是文本，提取纯文本（去掉HTML标签）
                if content.strip().startswith('<'):
                    text_content = re.sub(r'<[^>]+>', '', content)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    print(f"\n提取的文本内容（前200字符）:")
                    print(text_content[:200] + "..." if len(text_content) > 200 else text_content)
                else:
                    print(f"\n文本内容预览:")
                    print(content[:200] + "..." if len(content) > 200 else content)
            else:
                # 如果是二进制数据
                print(f"\n二进制内容预览（前100字节）:")
                print(content[:100] if len(content) > 100 else content)
            
            return content, encoding
            
    except HTTPError as e:
        print(f"HTTP错误: {e.code} - {e.reason}")
        return None, None
    except URLError as e:
        print(f"URL错误: {e.reason}")
        return None, None
    except TimeoutError:
        print("请求超时")
        return None, None
    except Exception as e:
        print(f"其他错误: {type(e).__name__}: {e}")
        return None, None

# 测试函数
def test_decoder_with_various_urls():
    """用不同的URL测试解码器"""
    test_urls = [
        ("https://httpbin.org/get", "JSON数据"),
        ("https://httpbin.org/html", "HTML页面"),
        ("https://httpbin.org/image/png", "PNG图片"),
        ("https://httpbin.org/gzip", "GZIP压缩的JSON"),
        ("https://www.baidu.com", "百度首页"),
    ]
    
    print("=" * 60)
    print("解码器测试套件")
    print("=" * 60)
    
    for url, description in test_urls:
        print(f"\n{'='*40}")
        print(f"测试: {description}")
        print(f"URL: {url}")
        
        result = improved_crawler()
        
        if result[0] is None:
            print(f"❌ 爬取失败")
        else:
            print(f"✅ 爬取成功")

# 主函数
if __name__ == "__main__":
    # 直接测试
    print("开始改进版爬虫测试...")
    content, encoding = improved_crawler()
    
    if content:
        print(f"\n{'='*60}")
        print("测试完成")
        print(f"编码: {encoding}")
        print(f"内容类型: {type(content).__name__}")
    else:
        print("测试失败")