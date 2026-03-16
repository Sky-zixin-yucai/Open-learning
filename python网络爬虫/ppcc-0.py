# ppcc-0.py学习
# 0.一次学习
from urllib import request

url = 'https://setuptools.pypa.io/en/latest/pkg_resources.html'# 包发现与资源访问pkg_resources
content = request.urlopen(url).read()

print(content)

## 问题1：没有设置请求头
" 1.很多网站会拒绝没有User-Agent的请求 ",
" 2.你得到的可能是403 Forbidden或者被重定向到验证页面 ",

## 问题2：没有错误处理
" 1.如果网站挂了、网络断了、URL错了 程序会直接崩溃 ",

## 问题3：编码问题
" 1.read()返回的是bytes，直接print会显示b'...'这样的字节串 ",
" 2.需要手动解码，但不知道用什么编码 "

## 问题4：没有超时设置
" 1.如果服务器不响应，程序会一直卡住",

# 对照 ppcc-1学习