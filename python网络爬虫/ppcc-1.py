""" 
raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 403: Forbidden

第1步：添加请求头和超时设置
解决的问题：防止被网站屏蔽，避免请求无响应卡死 

"""

from urllib import request
import ssl

# 修复SSL证书问题（某些网站需要）
ssl._create_default_https_context = ssl._create_unverified_context #小问题''

url = 'https://setuptools.pypa.io/en/latest/pkg_resources.html'

# 创建请求对象并添加请求头
req = request.Request(
    url,
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
)

# 添加超时设置（10秒）
content = request.urlopen(req, timeout=10).read()
print(content)

"1. 添加SSL证书处理 "
## import ssl
## ssl._create_default_https_context = ssl._create_unverified_context

"""

1.1 为什么需要这个？

某些网站（尤其是老网站）使用自签名或过期的SSL证书

如果不跳过验证，会抛出 SSL: CERTIFICATE_VERIFY_FAILED 错误

注意：这在生产环境中不安全，仅用于爬虫开发

"""

"""

1.2工作原理：

"""

# 默认情况下，Python会验证SSL证书
# 这行代码创建了一个不验证证书的SSL上下文

" ssl._create_default_https_context = ssl._create_unverified_context "

# 相当于告诉Python："不要检查网站的SSL证书是否有效"
# 这样即使证书有问题也能继续访问

"""

2.1 设置有效的URL

url = 'https://httpbin.org/get'

"""

# 选择httpbin.org的原因：

# 1.专门用于HTTP测试的服务

# 2.返回结构化的JSON数据，易于调试

# 3.对爬虫友好，不会封禁IP

# 4.免费且稳定

"""

3.1 创建Request对象并添加请求头

req = request.Request(
    url,
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
)

"""

"""

3.2 为什么要创建Request对象:

urlopen() 可以直接接受URL字符串 但无法添加自定义头部

Request 对象允许我们定制HTTP请求

"""

"""

4.1 请求头详解：

User-Agent 最重要

'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

作用：
1. 告诉服务器这是一个"正常浏览器"而不是爬虫
2. 不同User-Agent可能得到不同的响应（移动端/桌面端）
3. 没有User-Agent的请求容易被拒绝

"""

"""

4.2 为什么选择这个User-Agent？

- Windows 10 + Chrome浏览器的常见标识
- 被绝大多数网站接受

"""

"""

4.2 Accept 内容协商

'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'

"""

# 拆解：
# - text/html: 首选HTML格式
# - application/xhtml+xml: 其次接受XHTML+XML
# - application/xml: 再其次接受XML
# - q=0.9: 优先级（质量值），1.0最高
# - */*: 接受任何类型，优先级0.8

# 作用：告诉服务器客户端能接受哪些内容类型
# 模拟真实浏览器的行为

"""

5.1 发送请求并设置超时

content = request.urlopen(req, timeout=10).read()

关键改进 timeout=10

"""

# 原始代码的问题：
# request.urlopen(url).read()  # 没有超时，可能永远等待

# 修复后的代码：
# request.urlopen(req, timeout=10).read()

# timeout=10 表示：
# 1. 连接超时：建立TCP连接最多等10秒
# 2. 读取超时：从服务器读取数据最多等10秒
# 3. 超时后会抛出 `TimeoutError` 异常

"""

5.2 超时的重要性

"""

# 没有超时的情况：
# 1. 目标服务器宕机 → 程序永远等待 → 资源耗尽
# 2. 网络故障 → 程序卡住 → 需要手动终止

# 有超时的情况：
# 1. 目标服务器宕机 → 10秒后抛出异常 → 可以处理或重试
# 2. 网络故障 → 10秒后抛出异常 → 可以记录日志并继续

""" 5.3 print(content) """

# 对照 ppcc-2学习