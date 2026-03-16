"""

第2步 添加错误处理
解决的问题：防止网络问题或服务器错误导致程序崩溃

"""

"""
第二个修复要解决的核心问题

第一个修复遗留的问题：
无错误处理：任何异常都会导致程序崩溃

调试信息不足：不知道发生了什么，只知道程序停了

资源管理不当 没有关闭HTTP连接

第二个修复要解决的问题：
✅ 添加全面的异常处理：防止程序因网络问题崩溃

✅ 提供有用的调试信息：知道发生了什么错误，为什么失败

✅ 改进输出格式：添加状态码、响应头等信息

⚠️ 编码问题还未解决 仍然打印bytes 留到第三个修复

"""
from urllib import request
from urllib.error import URLError, HTTPError
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
url = 'https://setuptools.pypa.io/en/latest/pkg_resources.html'

try:
    req = request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
    )
    
    response = request.urlopen(req, timeout=10)
    content = response.read()
    
    print(f"状态码: {response.status}")
    print(f"响应头: {response.headers}")
    print(f"内容长度: {len(content)} 字节")
    print(content[:500])  # 只打印前500个字符
    
except HTTPError as e:
    print(f"HTTP错误: {e.code} - {e.reason}")
    print(f"错误URL: {e.url}")
    
except URLError as e:
    print(f"URL错误: {e.reason}")
    
except TimeoutError:
    print("请求超时，请检查网络或稍后重试")
    
except Exception as e:
    print(f"其他错误: {type(e).__name__}: {e}")

"""

1.1 导入专门的异常类

from urllib.error import URLError, HTTPError

为什么单独导入这些异常？

"""    

# 正确方式：
from urllib.error import URLError, HTTPError

# 为什么不用通用Exception？
# 因为我们需要区分不同类型的错误
# urllib.error提供了专门的异常类：

# HTTPError: HTTP协议错误（状态码400-599）
#   当服务器返回错误状态码时抛出
#   包含额外信息：.code（状态码）, .reason（原因）, .headers（响应头）

# URLError: URL相关错误
#   域名解析失败、网络不可达、连接被拒绝等
#   包含.reason属性

# TimeoutError: 超时错误（Python内置异常）
#   连接或读取超时

# OSError: 操作系统错误（父类）
#   URLError和TimeoutError都是OSError的子类

"""

1.2 使用try-except块包裹整个请求

"""

try:
    # 所有可能出错的代码放在这里
    req = request.Request(...)
    response = request.urlopen(...)
    content = response.read()
    # ... 处理成功的情况
    
except HTTPError as e:
    # 专门处理HTTP错误
    pass
    
except URLError as e:
    # 专门处理URL错误
    pass
    
except TimeoutError:
    # 专门处理超时
    pass
    
except Exception as e:
    # 兜底处理，捕获其他所有异常
    pass

"try-except的执行流程"

# 执行顺序：
# 1. 执行try块中的代码
# 2. 如果发生异常，Python停止执行try块剩余代码
# 3. Python检查异常类型，匹配第一个合适的except块
# 4. 执行匹配的except块中的代码
# 5. 如果没有匹配的except，异常继续向上传播

# 重要：except的顺序很重要！
# 应该从最具体的异常到最通用的异常

# ❌ 错误顺序：
"""

except Exception as e:  # 太宽泛，会捕获所有异常
    print("通用错误")
except HTTPError as e:   # 永远不会执行到这里！
    print("HTTP错误")

# ✅ 正确顺序：
except HTTPError as e:   # 先捕获最具体的
    print("HTTP错误")
except URLError as e:    # 次具体的
    print("URL错误")
except Exception as e:   # 最后用通用的兜底
    print("其他错误")
    
    """

"""

1.3 提取响应信息并格式化输出

"""

# 改进的响应处理
response = request.urlopen(req, timeout=10)

# 1. 读取响应内容
content = response.read()

# 2. 提取并格式化调试信息
print(f"状态码: {response.status}")
print(f"响应头: {response.headers}")
print(f"内容长度: {len(content)} 字节")
print(content[:500])  # 限制输出长度

"为什么要提取这些信息？"

# response.status - HTTP状态码
# 200: 成功
# 301/302: 重定向
# 403: 禁止访问（可能被反爬虫）
# 404: 页面不存在
# 500: 服务器内部错误
# 知道状态码就知道请求的基本结果

# response.headers - HTTP响应头
# Content-Type: text/html; charset=utf-8（编码信息）
# Content-Length: 12345（内容长度）
# Server: nginx（服务器类型）
# Set-Cookie: ...（设置cookies）
# 这些信息对调试非常重要

# len(content) - 内容长度
# 验证是否完整下载
# 如果长度异常小，可能被重定向到错误页面

# content[:500] - 只打印前500字符
# 避免在控制台输出大量内容
# 足够看到响应的大致结构

"""

每种异常的具体含义和场景

HTTPError详解

"""

# 触发条件：服务器返回4xx或5xx状态码
# 例如：404 Not Found, 403 Forbidden, 500 Internal Server Error
"""
except HTTPError as e:
    print(f"HTTP错误: {e.code} - {e.reason}")  # 状态码和原因短语
    print(f"错误URL: {e.url}")  # 引发错误的URL
    
    # e对象实际上是响应对象，可以读取内容
    try:
        error_content = e.read().decode('utf-8')
        print(f"错误页面内容: {error_content[:200]}...")
    except:
        pass
"""
# 实际例子：
# 访问 https://httpbin.org/status/404
# 会抛出 HTTPError: HTTP Error 404: NOT FOUND

"""

URLError详解

"""

# 触发条件：
# 1. 域名解析失败（DNS错误）
# 2. 网络不可达
# 3. 连接被拒绝（服务器没启动）
# 4. SSL/TLS错误
"""
except URLError as e:
    # e.reason 包含具体的错误信息
    error_reason = e.reason
    
    # reason可能是字符串，也可能是异常对象
    if isinstance(error_reason, str):
        print(f"URL错误: {error_reason}")
    else:
        # 如果是socket.error等异常
        print(f"URL错误: {error_reason.__class__.__name__}: {error_reason}")
"""        
    # 常见的URLError原因：
    # [Errno 8] nodename nor servname provided, or not known - 域名解析失败
    # [Errno 61] Connection refused - 连接被拒绝
    # [Errno 65] No route to host - 网络不可达


"""

TimeoutError详解

"""    

# 触发条件：请求超过设置的timeout时间
"""
except TimeoutError:
    # Python 3.10+中，urllib的超时抛出TimeoutError
    # 旧版本可能抛出socket.timeout或urllib.error.URLError
    
    print("请求超时，请检查网络或稍后重试")
"""
    # 超时的可能原因：
    # 1. 服务器响应太慢
    # 2. 网络连接质量差
    # 3. 服务器宕机
    # 4. 防火墙阻止

"""

通用的Exception兜底

"""    
"""
except Exception as e:
    print(f"其他错误: {type(e).__name__}: {e}")
"""
# 为什么需要这个兜底？
# 1. 捕获未预料到的异常类型
# 2. 防止程序完全崩溃
# 3. 提供最后的错误信息

# type(e).__name__ 获取异常类名
# 例如：MemoryError, ValueError, TypeError等

"""

执行流程示意图

开始执行
    ↓
创建Request对象（添加headers）
    ↓
发送HTTP请求（设置timeout=10秒）
    ├── 成功（服务器响应） → 进入成功流程
    ├── HTTP错误（4xx/5xx） → HTTPError → 对应处理
    ├── URL错误（网络/DNS） → URLError → 对应处理
    ├── 超时（>10秒） → TimeoutError → 对应处理
    └── 其他错误 → Exception → 对应处理

成功流程：
    读取响应内容
    输出状态码、响应头、内容长度
    打印内容前500字符
    结束

错误处理流程：
    根据异常类型输出对应错误信息
    结束（不崩溃）

"""