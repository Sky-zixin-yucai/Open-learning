"""

第3步 处理编码问题

解决的问题 避免bytes直接显示为b'...'，正确处理中文和其他编码

第三个修复代码的核心目标
解决前两个修复遗留的最大问题：编码处理

# 第二个修复的遗留问题：
print(content[:500])  # content是bytes类型 输出 b'...\xe4\xb8\xad\xe6\x96\x87...'

要解决的具体问题：
字节到字符串的转换：如何正确解码 bytes 到 str

编码自动检测：不知道网页使用什么编码时怎么办

中文乱码问题：为什么中文显示为 \xe4\xb8\xad\xe6\x96\x87

内容智能显示 如何区分HTML和纯文本

"""

from urllib import request
from urllib.error import URLError, HTTPError
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
url = 'https://setuptools.pypa.io/en/latest/pkg_resources.html'  # 可以换成中文网站测试

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
        
        # 读取原始字节
        content_bytes = response.read()
        
        # 获取响应头中的编码信息
        content_type = response.headers.get('Content-Type', '')
        encoding = None
        
        # 从Content-Type中提取编码
        if 'charset=' in content_type:
            encoding = content_type.split('charset=')[-1].split(';')[0].strip()
        
        # 如果没有指定编码，尝试自动检测
        if encoding is None:
            # 尝试常见编码
            for enc in ['utf-8', 'gbk', 'gb2312', 'iso-8859-1', 'big5']:
                try:
                    content = content_bytes.decode(enc)
                    encoding = enc
                    print(f"自动检测到编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # 所有编码都失败，使用替代方案
                content = content_bytes.decode('utf-8', errors='replace')
                encoding = 'utf-8 (有替代字符)'
                print(f"无法确定编码，使用UTF-8并替换无法解码的字符")
        else:
            # 使用指定的编码
            try:
                content = content_bytes.decode(encoding)
            except UnicodeDecodeError:
                # 如果指定的编码不对，回退到UTF-8
                content = content_bytes.decode('utf-8', errors='replace')
                encoding = 'utf-8 (回退)'
                print(f"指定编码{encoding}解码失败，回退到UTF-8")
        
        print(f"最终使用的编码: {encoding}")
        print(f"内容长度: {len(content)} 字符")
        
        # 智能显示内容：如果是HTML，提取文本；否则直接显示
        if 'text/html' in content_type or content.strip().startswith('<'):
            # 简单提取HTML中的文本（去掉标签）
            import re
            text_content = re.sub(r'<[^>]+>', '', content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            print(f"提取的文本内容（前200字符）: {text_content[:200]}...")
        else:
            # 显示前200个字符
            print(f"内容预览: {content[:200]}...")
            
except HTTPError as e:
    print(f"HTTP错误: {e.code} - {e.reason}")
    
except URLError as e:
    print(f"URL错误: {e.reason}")
    
except TimeoutError:
    print("请求超时")
    
except Exception as e:
    print(f"其他错误: {type(e).__name__}: {e}")



print("==============================================================")
    

"""

改进1 新增请求头 - 更模拟真实浏览器

'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
'Accept-Encoding': 'gzip, deflate, br'

"""    

"为什么添加这些头？"

"""

1.1 Accept-Language

# 告诉服务器用户偏好的语言
'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'

# 解析：
# zh-CN: 简体中文 优先级0.9
# zh: 中文 通用  优先级0.9
# en: 英文 优先级0.8

# 作用：
# 1. 中文网站可能返回中文内容
# 2. 多语言网站返回对应语言版本
# 3. 更真实地模拟浏览器行为

"""

"""

1.2 Accept-Encoding

# 告诉服务器客户端支持的内容压缩方式
'Accept-Encoding': 'gzip, deflate, br'

# 压缩方式：
# gzip: 最常用的压缩格式
# deflate: 另一种压缩格式
# br: Brotli压缩 较新 压缩率更高

# 作用：
# 1. 减少数据传输量
# 2. 服务器可能返回压缩后的内容
# 3. Python的urllib会自动解压gzip和deflate

# ⚠️ 注意：
# urllib默认不支持brotli(br)压缩
# 如果服务器返回br压缩 需要额外处理

"""

"""

改进2 使用with语句 - 资源自动管理

"""

with request.urlopen(req, timeout=10) as response:
    # 在这个代码块中操作response
    content_bytes = response.read()
    # ...
# with语句结束后，response自动关闭

# 对比旧的写法：
response = request.urlopen(req, timeout=10)  # 需要手动关闭
content = response.read()
# 可能忘记关闭连接

# with语句的工作原理：
# 1. 调用request.urlopen()返回一个上下文管理器对象
# 2. 进入with块时，执行__enter__()方法
# 3. 退出with块时，执行__exit__()方法（关闭连接）

"""

改进3 获取Content-Type头 - 理解内容类型

"""

content_type = response.headers.get('Content-Type', '')

# Content-Type的常见值：
# 1. text/html; charset=utf-8     - HTML页面，UTF-8编码
# 2. application/json             - JSON数据
# 3. text/plain; charset=gbk      - 纯文本，GBK编码
# 4. application/xml              - XML数据
# 5. image/jpeg                   - JPEG图片

"""

改进4 提取编码信息 - 关键步骤

"""

# 方法1：从Content-Type中提取
if 'charset=' in content_type:
    encoding = content_type.split('charset=')[-1].split(';')[0].strip()

# 举例：
# Content-Type: text/html; charset=utf-8
# 提取过程：
# 1. 找到 'charset=' 的位置
# 2. 取 'charset=' 后面的部分：'utf-8'
# 3. 如果有分号，去掉分号后的部分
# 4. 去掉空白字符

"""

改进5 编码自动检测 - 智能解码

"""

# 如果没有指定编码，尝试常见编码
for enc in ['utf-8', 'gbk', 'gb2312', 'iso-8859-1', 'big5']:
    try:
        content = content_bytes.decode(enc)
        encoding = enc
        print(f"自动检测到编码: {encoding}")
        break  # 成功解码，跳出循环
    except UnicodeDecodeError:
        continue  # 尝试下一个编码

# 这个循环的工作方式：
# 1. 首先尝试utf-8（最常用）
# 2. 如果失败，尝试gbk（中文Windows常用）
# 3. 然后尝试gb2312（简体中文标准）
# 4. 再尝试iso-8859-1（西欧语言）
# 5. 最后尝试big5（繁体中文）

"""

改进6 错误处理和回退机制

# 情况1 所有编码都尝试失败
else:  # 注意 这个else是for循环的else
    content = content_bytes.decode('utf-8', errors='replace')
    encoding = 'utf-8 (有替代字符)'

# errors='replace'参数：
# 无法解码的字节会被替换为 �（替换字符）
# 这样至少能得到部分可读内容，而不是直接崩溃

# 情况2 指定的编码解码失败
try:
    content = content_bytes.decode(encoding)
except UnicodeDecodeError:
    content = content_bytes.decode('utf-8', errors='replace')
    encoding = 'utf-8 (回退)'

# 这种回退机制确保：
# 1. 即使编码信息有误，程序也不会崩溃
# 2. 用户至少能看到部分内容
# 3. 程序可以继续执行

"""

"""

改进7 智能内容显示

"""

# 判断是否是HTML
if 'text/html' in content_type or content.strip().startswith('<'):
    # 提取文本内容
    text_content = re.sub(r'<[^>]+>', '', content)  # 去掉HTML标签
    text_content = re.sub(r'\s+', ' ', text_content).strip()  # 合并空白字符
    print(f"提取的文本内容（前200字符）: {text_content[:200]}...")
else:
    # 非HTML内容直接显示
    print(f"内容预览: {content[:200]}...")

# 正则表达式详解：
# r'<[^>]+>' 匹配所有HTML标签
# <     : 匹配开始尖括号
# [^>]+ : 匹配一个或多个非>字符
# >     : 匹配结束尖括号
# 这样 <div>、</p>、<br /> 等都会被匹配

# 为什么这么做？
# HTML源码：<h1>标题</h1><p>内容</p>
# 提取后：标题 内容
# 更易读，更适合调试


## 2.编码问题的深层原理


"2.1 问题：为什么会有编码问题？"

# 根源：计算机存储的是二进制（bytes），但人类需要文字（str）

# 过程：
# 1. 服务器：字符串 → 编码 → 字节流 → 网络传输
# 2. 客户端：接收字节流 → 解码 → 字符串

# 如果编码和解码使用的规则不一致，就出现乱码

# 常见编码：
# UTF-8: 全球通用，一个汉字3个字节
# GBK: 中文Windows常用，一个汉字2个字节
# ASCII: 英文字母，1个字节

# 创建包含中文的字符串
chinese_text = "你好，世界！"

# 用不同编码转换成bytes
utf8_bytes = chinese_text.encode('utf-8')  # b'\xe4\xbd\xa0\xe5\xa5\xbd...'
gbk_bytes = chinese_text.encode('gbk')     # b'\xc4\xe3\xba\xc3...'

print(f"UTF-8编码: {utf8_bytes}")
print(f"GBK编码: {gbk_bytes}")

# 用错误编码解码会怎样？
try:
    wrong_text = utf8_bytes.decode('gbk')  # 错误：用GBK解码UTF-8
    print(f"错误解码: {wrong_text}")
except UnicodeDecodeError as e:
    print(f"解码错误: {e}")

# 用正确编码解码
correct_text = utf8_bytes.decode('utf-8')
print(f"正确解码: {correct_text}")

"""

HTTP响应中的编码信息流程

服务器生成内容 → 选择编码 → 设置Content-Type头 → 发送
                    ↓
客户端接收 → 读取Content-Type → 用指定编码解码 → 显示

"""
