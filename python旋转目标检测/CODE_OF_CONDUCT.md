# Learning Journey - 学习之旅

>>你会记住你走过的路、看过的事、见过的人。
>>You will remember the paths you've taken, the things you've seen, and the people you've met.


> Progressive learning project, documenting learning journeys from scratch  
> 渐进式学习项目，记录从零开始的学习旅程

## Project Introduction | 项目简介

This is an open-source project that records real learning processes. We start with a completely unfamiliar technology and fully document every step from encountering errors to solving problems.  
这是一个记录真实学习过程的开源项目。我们从一个完全陌生的技术开始，完整记录从遇到错误到解决问题的每一步。

## Learning Modules | 学习模块

Currently, we have the following learning modules:  
目前，我们有以下学习模块：

1. [Python Web Crawler](学习/python网络爬虫/) - From HTTP 403 errors to handling Brotli compression  
   [Python网络爬虫](学习/python网络爬虫/) - 从HTTP 403错误到处理Brotli压缩

2. [Python Rotated Object Detection](学习/python旋转目标检测/) - A journey of debugging file operations  
   [Python旋转目标检测](学习/python旋转目标检测/) - 一次文件操作调试之旅

Each module is organized in a progressive learning style. You can experience the entire learning process by reading and running the code.  
每个模块都按照渐进式学习的方式组织，你可以通过阅读和运行代码来体验整个学习过程。

### Python Rotated Object Detection | Python旋转目标检测

This module records a debugging journey for file operations in a rotated object detection project.

#### Stage 1: dota-0.py
- **Status**: Runtime failure.
- **Description**: 在处理python旋转目标检测，这个项目的时候。我的代码在dota-0.py运行时，找不到我的文件路径。  
  When working on the Python rotated object detection project, my code in dota-0.py couldn't find my file path during runtime.

#### Stage 2: dota-0.1.py
- **Status**: Apparent success, actual failure.
- **Description**: 在处理python旋转目标检测，这个项目的时候。我的代码在dota-0.1.py运行时，文件读取成功，创建了保存文件，但搞笑的是，根本就没有保存文件，也根本没有读取成功。  
  When working on the Python rotated object detection project, my code in dota-0.1.py successfully read the file and created a save file during runtime, but the funny thing is, there was actually no save file created, and the file reading wasn't successful at all.

#### Stage 3: dota-0-2.py
- **Status**: Runtime success.
- **Description**: 在处理python旋转目标检测，这个项目的时候。我的代码在dota-0-2.py运行时，文件读取成功并创建了保存文件，这回终于成功了。  
  When working on the Python rotated object detection project, my code in dota-0-2.py successfully read the file and created a save file during runtime, and this time it finally worked.

## How to Use? | 如何使用？

1. Choose a technical field that interests you (see the learning modules above).  
   选择一个你感兴趣的技术领域（如上方的学习模块）。

2. Enter the corresponding directory and read README.md to understand the learning objectives of that module.  
   进入对应的目录，阅读README.md了解该模块的学习目标。

3. Run the code in the order of file numbers (e.g., `ppcc-0.py`, `ppcc-1.py`...) and observe the errors and solutions.  
   按照文件编号顺序（如`ppcc-0.py`, `ppcc-1.py`...）运行代码，观察错误和解决方案。

4. Understand the principles of each step through code comments and documentation.  
   通过代码注释和文档理解每个步骤的原理。

## Contribution | 贡献

We welcome new learning modules or improvements to existing modules. Before contributing, please read:  
我们欢迎新的学习模块或对现有模块的改进。贡献前请阅读：

- [Contribution Guidelines](CONTRIBUTING.md) | [贡献指南](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md) | [行为准则](CODE_OF_CONDUCT.md)
- [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md) | [贡献者协议](CONTRIBUTOR_AGREEMENT.md)

## License | 许可证

This project uses the Apache 2.0 license. See the [LICENSE](LICENSE) file for details.  
本项目采用Apache 2.0许可证，详见[LICENSE](LICENSE)文件。

## Contact | 联系

If you have questions, you can contact us via GitHub Issues or email (if provided).  
如有问题，可以通过GitHub Issues或邮件（如果有提供）联系。

---

*Learning is not the destination, but the starting point.*  
*学习不是终点，而是起点。*