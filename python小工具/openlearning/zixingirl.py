import builtins
original_print = builtins.print
nome = "你的python版本是: " + str(__import__('sys').version_info.major) + "." + str(__import__('sys').version_info.minor)
original_print(nome)
print_print = "你要参与哪个子项目？"
print("====="*10)
print("1. python网络爬虫")
print("====="*10)
print("2. python旋转目标检测")
print("====="*10)
print("3. Rule-Governed Architecture")
print("====="*10)
original_print(print_print)
choice = input("请输入数字1、2或3选择子项目: ")
if choice == '1':
      print("====="*10)
      original_print("请注意，你需要安装brotli库")
      print("你可以使用以下命令安装brotli库:")
      print("====="*10)
      print("pip install brotli")
      print("====="*10)
elif choice == '2':
      original_print("请注意，你需要安装opencv-python库")
      print("你可以使用以下命令安装opencv-python库:")
      print("====="*10)
      print("pip install opencv-python")
      print("====="*10)
elif choice == '3':
      original_print("你选择了Rule-Governed Architecture子项目！")
      print("请按照项目说明进行操作。")
      print("====="*10)
      original_print("""Rule-Governed Architecture子项目旨在开发一个基于规则的系统架构""")
      choice = input("请输入数字1、2或3进行选择安装: ")
      if choice == '1':
          original_print("你选择了安装依赖项！")
          print("请使用以下命令安装所需的依赖项:")
          print("====="*10)
          print("pip install -r requirements.txt")
          print("====="*10)
      elif choice == '2':
          original_print("你选择了查看项目文档！")
          print("请访问以下链接查看项目文档:")
          print("====="*10)
          print("https://github.com/Sky-zixin-yucai/Open-learning/releases/tag/vv1.0.0")
          print("====="*10)
      elif choice == '3':
          original_print("你选择了运行示例代码！")
          print("请按照项目说明运行示例代码。")
          print("====="*10)
      else:
          original_print("无效选择，请输入1、2或3。")
else:
      original_print("无效选择，请输入1、2或3。")
