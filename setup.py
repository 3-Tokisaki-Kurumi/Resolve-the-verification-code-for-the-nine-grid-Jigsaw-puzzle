from setuptools import setup, find_packages

# 读取README文件内容作为长描述 | Read README file content as long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # 包名称 | Package name
    name="nine-grid-captcha-solver",
    # 版本号 | Version number
    version="0.1.0",
    # 作者信息 | Author information
    author="Tokisaki Kurumi",
    author_email="your.email@example.com",
    # 包简短描述 | Package short description
    description="九宫格拼图验证码自动求解工具 | Nine-Grid Jigsaw Captcha Solver",
    # 包详细描述 | Package detailed description
    long_description=long_description,
    long_description_content_type="text/markdown",
    # 项目主页 | Project homepage
    url="https://github.com/3-Tokisaki-Kurumi/Resolve-the-verification-code-for-the-nine-grid-Jigsaw-puzzle",
    # 包含的包 | Included packages
    packages=find_packages(),
    # 分类标签 | Classification tags
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Security",
        "Topic :: Multimedia :: Graphics",
    ],
    # Python版本要求 | Python version requirement
    python_requires=">=3.6",
    # 依赖包 | Dependencies
    install_requires=[
        "requests>=2.25.0",
        "pillow>=8.0.0",
        "numpy>=1.19.0",
        "beautifulsoup4>=4.9.0",
        "matplotlib>=3.3.0",
    ],
    # 命令行入口点 | Command line entry points
    entry_points={
        "console_scripts": [
            "ngcaptcha-solve=nine_grid_captcha_solver.cli:main",
        ],
    },
    # 包含额外文件 | Include extra files
    include_package_data=True,
) 
