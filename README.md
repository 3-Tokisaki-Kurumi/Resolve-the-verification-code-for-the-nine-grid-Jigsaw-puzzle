# 九宫格验证码破解 | Nine-Grid Captcha Solver

一个基于图像处理和边缘特征匹配的九宫格拼图验证码自动求解工具。无需训练，传统算法，准确率超过95%！

A tool for automatically solving nine-grid jigsaw captchas based on image processing and edge feature matching.No need for training, traditional algorithms have an accuracy rate of over 95%!

可以与自动化工具比如selenium等结合使用

Can be used in conjunction with automation tools such as selenium

![示例结果 | Example Result](In folder test_original and test_result)

## 项目简介 | Project Introduction

九宫格拼图验证码是一种常见的人机验证方式，它将一张完整图片分割成9个部分并打乱顺序，要求用户将其还原为原始图像。本项目通过分析图像边缘特征、计算相似度，自动寻找最优拼图排列方案，从而实现验证码的自动求解。

The nine-grid jigsaw captcha is a common human-machine verification method that divides a complete image into 9 parts and scrambles their order, requiring users to restore them to the original image. This project automatically solves such captchas by analyzing image edge features, calculating similarities, and finding the optimal arrangement.

### 核心功能 | Core Features

- 从HTML页面或URL中提取九宫格验证码图片 | Extract nine-grid captcha images from HTML pages or URLs
- 通过边缘特征匹配算法自动分析图片拼接关系 | Automatically analyze image connections using edge feature matching algorithms
- 生成最少步骤的交换序列 | Generate minimal swap sequences
- 支持本地图片测试和可视化展示 | Support local image testing and visualization
- 高效启发式搜索算法，结合贪心和回溯策略 | Efficient heuristic search algorithm combining greedy and backtracking strategies
- 并行下载和处理图片 | Parallel downloading and processing of images

## 安装方法 | Installation

从GitHub安装：

Install from GitHub:

```bash
pip install git+https://github.com/3-Tokisaki-Kurumi/Resolve-the-verification-code-for-the-nine-grid-Jigsaw-puzzle.git
```

## 使用方法 | Usage

### 作为命令行工具使用 | As Command Line Tool

安装后可直接在命令行中使用：

After installation, you can use it directly in the command line:

```bash
# 解析本地图片文件夹 | Parse local image folder
ngcaptcha-solve --local /path/to/images

# 解析HTML文件 | Parse HTML file
ngcaptcha-solve --html /path/to/file.html

# 解析URL | Parse URL
ngcaptcha-solve --url "https://example.com/captcha-page"
```

### 作为Python库使用 | As Python Library

```python
from nine_grid_captcha_solver import CaptchaSolver

# 初始化求解器 | Initialize solver
solver = CaptchaSolver()

# 方法1：从本地图片测试 | Method 1: Test from local images
swaps, grid = solver.test_local_images("test_images_folder")

# 方法2：从HTML文件解析 | Method 2: Parse from HTML file
swaps, grid = solver.solve_from_file("captcha_page.html")

# 方法3：从URL解析 | Method 3: Parse from URL
swaps, grid = solver.solve_from_url("https://example.com/captcha-page")

# 输出结果 | Output results
if swaps:
    print("交换步骤 | Swap steps:", swaps)
    print("最终网格排列 | Final grid arrangement:", grid)
    
    # 可视化交换过程 | Visualize swap process
    solver.simulate_swaps(swaps)
```

## 工作原理 | How It Works

1. **图像预处理 | Image Preprocessing**: 将九宫格拼图图片转换为灰度图像 | Convert nine-grid jigsaw images to grayscale
2. **边缘特征提取 | Edge Feature Extraction**: 提取每张图片的四条边缘的灰度特征 | Extract grayscale features from all four edges of each image
3. **相似度计算 | Similarity Calculation**: 计算边缘之间的余弦相似度矩阵 | Calculate cosine similarity matrices between edges
4. **最优排列搜索 | Optimal Arrangement Search**: 使用启发式算法搜索最佳拼图排列 | Use heuristic algorithms to search for the best jigsaw arrangement
5. **交换序列生成 | Swap Sequence Generation**: 计算从初始状态到目标排列的最少交换步骤 | Calculate minimum swap steps from initial state to target arrangement

## 性能指标 | Performance

- 平均识别准确率：超过95%（取决于图片清晰度和特征显著性）| Average recognition accuracy: over 95% (depends on image clarity and feature significance)
- 平均解决时间：0.5~2秒（根据硬件性能有所差异）| Average solving time: 0.5-2 seconds (varies with hardware performance)

## 示例 | Example

```python
from nine_grid_captcha_solver import CaptchaSolver

solver = CaptchaSolver()
folder = "test_original"
swaps, grid = solver.test_local_images(folder)

if swaps:
    print("\n交换步骤 | Swap steps:", swaps)
    print("最终网格排列 | Final grid arrangement:")
    for i, row in enumerate(grid):
        print(f"行 | Row {i+1}: {row}")
    
    # 模拟交换过程 | Simulate swap process
    solver.simulate_swaps(swaps)
```

## 贡献指南 | Contribution Guidelines

欢迎提交问题报告和改进建议！如果您想贡献代码，请遵循以下步骤：

Contributions, issues, and feature requests are welcome! Follow these steps:

1. Fork 本仓库 | Fork this repository
2. 创建您的特性分支 | Create your feature branch (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 | Commit your changes (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 | Push to the branch (`git push origin feature/amazing-feature`)
5. 打开一个 Pull Request | Open a Pull Request

## 许可证 | License

本项目采用 MIT 许可证 - 详情见 [LICENSE](LICENSE) 文件

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details 
