"""
九宫格验证码求解器命令行接口
Command-line interface for Nine-Grid Captcha Solver
"""

import argparse
import sys
import logging
from .solver import CaptchaSolver

def main():
    """
    命令行主入口函数
    Main entry function for command line
    """
    parser = argparse.ArgumentParser(description="九宫格拼图验证码自动求解工具 | Nine-Grid Jigsaw Captcha Solver")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--local", metavar="FOLDER", help="从本地图片文件夹解析验证码 | Parse captcha from local image folder")
    group.add_argument("--html", metavar="FILE", help="从HTML文件解析验证码 | Parse captcha from HTML file")
    group.add_argument("--url", metavar="URL", help="从URL解析验证码 | Parse captcha from URL")
    
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="日志级别 | Log level")
    parser.add_argument("--timeout", type=int, default=5, help="求解超时时间（秒） | Solving timeout (seconds)")
    parser.add_argument("--match-id", help="可选的匹配ID，用于缓存 | Optional match ID for caching")
    parser.add_argument("--visualize", action="store_true", help="是否显示可视化结果 | Whether to show visualization results")
    
    args = parser.parse_args()
    
    # 设置日志级别 | Set log level
    log_level = getattr(logging, args.log_level)
    
    # 初始化求解器 | Initialize solver
    solver = CaptchaSolver(log_level=log_level, timeout=args.timeout)
    
    try:
        if args.local:
            print(f"从本地图片文件夹解析验证码 | Parsing captcha from local folder: {args.local}")
            swaps, grid = solver.test_local_images(args.local)
        elif args.html:
            print(f"从HTML文件解析验证码 | Parsing captcha from HTML file: {args.html}")
            swaps, grid = solver.solve_from_file(args.html, args.match_id)
        elif args.url:
            print(f"从URL解析验证码 | Parsing captcha from URL: {args.url}")
            swaps, grid = solver.solve_from_url(args.url, args.match_id)
        
        if swaps:
            print("\n交换步骤 | Swap steps:", swaps)
            print("最终网格排列 | Final grid arrangement:")
            for i, row in enumerate(grid):
                print(f"行 | Row {i+1}: {row}")
            
            # 模拟交换过程 | Simulate swap process
            print("\n模拟交换过程 | Simulating swap process:")
            solver.simulate_swaps(swaps)
            
            print("\n验证码破解成功！| Captcha solving successful!")
            return 0
        else:
            print("验证码破解失败 | Captcha solving failed")
            return 1
            
    except Exception as e:
        print(f"发生错误 | Error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 