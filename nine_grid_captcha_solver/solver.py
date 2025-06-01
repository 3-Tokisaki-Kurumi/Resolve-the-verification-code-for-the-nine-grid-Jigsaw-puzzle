"""
九宫格验证码求解器核心类
Core class for Nine-Grid Captcha Solver
"""

import requests
from PIL import Image
from io import BytesIO
import numpy as np
import itertools
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import signal
import os
import logging
import glob
import matplotlib.pyplot as plt

# 配置日志 | Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CaptchaSolver')

class TimeoutException(Exception):
    """
    超时异常类
    Timeout exception class
    """
    pass

class CaptchaSolver:
    """
    九宫格验证码求解器
    Nine-Grid Captcha Solver
    """
    
    def __init__(self, log_level=logging.INFO, timeout=5):
        """
        初始化验证码求解器
        Initialize the captcha solver
        
        Args:
            log_level: 日志级别 | Log level
            timeout: 求解超时时间（秒） | Solving timeout (seconds)
        """
        self.logger = logger
        self.logger.setLevel(log_level)
        self.timeout = timeout
        
        # 缓存 | Cache
        self.feature_cache = {}
        self.image_cache = {}
    
    def timeout_handler(self, signum, frame):
        """
        超时处理函数
        Timeout handler function
        """
        raise TimeoutException("Captcha solving timed out")
    
    def download_image(self, url):
        """
        下载单个图片并缓存
        Download a single image and cache it
        
        Args:
            url: 图片URL | Image URL
        
        Returns:
            PIL.Image: 下载的图片对象 | Downloaded image object
        """
        if url in self.image_cache:
            return self.image_cache[url]
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            self.image_cache[url] = img
            return img
        except Exception as e:
            self.logger.error(f"下载图片失败 | Image download failed: {url} - {str(e)}")
            return None
    
    def download_images(self, html_content, match_id=None):
        """
        从HTML内容提取图片URL并并行下载图片
        Extract image URLs from HTML content and download images in parallel
        
        Args:
            html_content: HTML内容 | HTML content
            match_id: 可选的匹配ID，用于缓存 | Optional match ID for caching
        
        Returns:
            list: 图片对象列表 | List of image objects
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        img_tags = soup.select('.grid-item img')
        
        if not img_tags:
            self.logger.error("未找到图片元素 | No image elements found")
            return []
        
        urls = []
        for img in img_tags:
            src = img.get('src', '')
            if not src:
                continue
                
            if src.startswith('//'):
                url = 'https:' + src
            elif src.startswith('/'):
                url = 'https://www.internationalsaimoe.moe' + src
            else:
                url = src
                
            # 添加时间戳防止缓存
            if '?' in url:
                url += f"&t={int(time.time())}"
            else:
                url += f"?t={int(time.time())}"
                
            urls.append(url)
        
        # 并行下载
        images = []
        with ThreadPoolExecutor(max_workers=9) as executor:
            future_to_url = {executor.submit(self.download_image, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    img = future.result()
                    if img:
                        images.append(img)
                except Exception as e:
                    self.logger.error(f"图片下载失败 | Image download failed: {url} - {str(e)}")
        
        return images
    
    def extract_edges(self, img, edge_size=20):
        """
        提取图片四边特征并压缩
        Extract and compress the features of all four edges of an image
        
        Args:
            img: PIL图像对象 | PIL image object
            edge_size: 压缩后的边缘特征尺寸 | Compressed edge feature size
        
        Returns:
            tuple: (左边缘, 右边缘, 上边缘, 下边缘)的特征向量 | Feature vectors for (left edge, right edge, top edge, bottom edge)
        """
        if img.mode != 'L':
            gray = img.convert('L')
        else:
            gray = img
            
        width, height = gray.size
        
        # 提取四条边缘
        left_edge = np.array([gray.getpixel((0, y)) for y in range(height)])
        right_edge = np.array([gray.getpixel((width-1, y)) for y in range(height)])
        top_edge = np.array([gray.getpixel((x, 0)) for x in range(width)])
        bottom_edge = np.array([gray.getpixel((x, height-1)) for x in range(width)])
        
        # 压缩边缘特征
        def compress_edge(edge, new_size):
            step = len(edge) / new_size
            return [np.mean(edge[int(i*step):int((i+1)*step)]) for i in range(new_size)]
        
        return (
            compress_edge(left_edge, edge_size),
            compress_edge(right_edge, edge_size),
            compress_edge(top_edge, edge_size),
            compress_edge(bottom_edge, edge_size)
        )
    
    def precompute_similarities(self, edge_features):
        """
        预计算相似度矩阵
        Precompute similarity matrices
        
        Args:
            edge_features: 边缘特征列表 | List of edge features
        
        Returns:
            tuple: (水平相似度矩阵, 垂直相似度矩阵) | (Horizontal similarity matrix, Vertical similarity matrix)
        """
        n = len(edge_features)
        
        # 提取所有边缘特征
        rights = np.array([feat[1] for feat in edge_features])
        lefts = np.array([feat[0] for feat in edge_features])
        bottoms = np.array([feat[3] for feat in edge_features])
        tops = np.array([feat[2] for feat in edge_features])
        
        # 向量化计算余弦相似度
        def vectorized_cosine(A, B):
            dot = np.dot(A, B.T)
            norm_A = np.linalg.norm(A, axis=1, keepdims=True)
            norm_B = np.linalg.norm(B, axis=1, keepdims=True)
            return dot / (norm_A * norm_B.T + 1e-8)
        
        H = vectorized_cosine(rights, lefts)  # 水平相似度
        V = vectorized_cosine(bottoms, tops)  # 垂直相似度
        
        # 将对角线设为最小值（避免自匹配）
        np.fill_diagonal(H, -10)
        np.fill_diagonal(V, -10)
        
        return H, V
    
    def optimized_search(self, H, V):
        """
        使用贪心+回溯的启发式搜索最优排列
        Use greedy+backtracking heuristic search to find the optimal arrangement
        
        Args:
            H: 水平相似度矩阵 | Horizontal similarity matrix
            V: 垂直相似度矩阵 | Vertical similarity matrix
        
        Returns:
            list: 最优排列的网格，3x3的二维列表 | Optimal arrangement grid, 3x3 2D list
        """
        n = H.shape[0]
        best_score = -float('inf')
        best_grid = None
        
        # 尝试不同的起始点
        for start in range(n):
            used = set([start])
            grid = [[start]]
            success = True
            
            # 构建第一行
            for col in range(1, 3):
                last = grid[0][-1]
                
                # 获取候选列表，按相似度降序排列
                candidates = np.argsort(H[last])[::-1]
                found = False
                
                for cand in candidates:
                    if cand not in used and H[last, cand] > 0.1:  # 相似度阈值
                        grid[0].append(cand)
                        used.add(cand)
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if not success:
                continue
            
            # 构建后续行
            for row in range(1, 3):
                grid.append([])
                for col in range(3):
                    if row == 0:
                        above = None
                    else:
                        above = grid[row-1][col]
                    
                    # 获取候选列表
                    if above is not None:
                        candidates = np.argsort(V[above])[::-1]
                    else:
                        candidates = list(range(n))
                    
                    found = False
                    for cand in candidates:
                        if cand not in used:
                            # 检查水平一致性（如果左边有图片）
                            if col > 0:
                                left = grid[row][col-1]
                                if H[left, cand] < 0.1:  # 水平相似度阈值
                                    continue
                            
                            # 检查垂直一致性
                            if above is not None and V[above, cand] < 0.1:
                                continue
                                
                            grid[row].append(cand)
                            used.add(cand)
                            found = True
                            break
                    
                    if not found:
                        success = False
                        break
                        
                if not success:
                    break
            
            if not success or len(used) != n:
                continue
                
            # 评估当前网格
            score = self.evaluate_grid(grid, H, V)
            if score > best_score:
                best_score = score
                best_grid = [row[:] for row in grid]
        
        return best_grid
    
    def evaluate_grid(self, grid, H, V):
        """
        评估网格得分
        Evaluate grid score
        
        Args:
            grid: 网格排列 | Grid arrangement
            H: 水平相似度矩阵 | Horizontal similarity matrix
            V: 垂直相似度矩阵 | Vertical similarity matrix
        
        Returns:
            float: 总相似度得分 | Total similarity score
        """
        score = 0
        # 水平连接
        for row in grid:
            for i in range(len(row)-1):
                score += H[row[i], row[i+1]]
        
        # 垂直连接
        for col in range(len(grid[0])):
            for i in range(len(grid)-1):
                score += V[grid[i][col], grid[i+1][col]]
        
        return score
    
    def generate_swap_sequence(self, initial, correct):
        """
        生成交换步骤
        Generate swap steps
        
        Args:
            initial: 初始序列 | Initial sequence
            correct: 正确序列 | Correct sequence
        
        Returns:
            list: 交换步骤列表，每步为一个(索引1, 索引2)元组 | List of swap steps, each step is a (index1, index2) tuple
        """
        swaps = []
        current = initial.copy()
        positions = {val: idx for idx, val in enumerate(current)}
        
        for idx in range(len(correct)):
            if current[idx] == correct[idx]:
                continue
                
            # 找到需要交换的位置
            swap_idx = positions[correct[idx]]
            
            # 执行交换
            current[idx], current[swap_idx] = current[swap_idx], current[idx]
            positions[current[idx]] = idx
            positions[current[swap_idx]] = swap_idx
            
            # 记录交换步骤
            swaps.append((idx, swap_idx))
        
        return swaps
    
    def solve_captcha(self, html_content, match_id=None):
        """
        主函数：解决验证码
        Main function: Solve captcha
        
        Args:
            html_content: HTML内容 | HTML content
            match_id: 可选的匹配ID，用于缓存 | Optional match ID for caching
        
        Returns:
            tuple: (交换步骤列表, 最终网格排列) | (List of swap steps, final grid arrangement)
        """
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            start_time = time.time()
            
            # 1. 下载图片 | Download images
            self.logger.info("开始下载图片... | Starting to download images...")
            images = self.download_images(html_content, match_id)
            
            if len(images) != 9:
                self.logger.error(f"图片数量错误: 需要9张, 实际获取{len(images)}张 | Image count error: Need 9, got {len(images)}")
                signal.alarm(0)
                return self.fallback_strategy()
            
            # 保存原始图片用于调试 | Save original images for debugging
            self.save_images(images, folder="original_images")
            
            # 2. 提取边缘特征 | Extract edge features
            self.logger.info("提取边缘特征... | Extracting edge features...")
            edge_features = []
            for i, img in enumerate(images):
                cache_key = f"{match_id}_{i}" if match_id else None
                if cache_key and cache_key in self.feature_cache:
                    features = self.feature_cache[cache_key]
                else:
                    features = self.extract_edges(img)
                    if cache_key:
                        self.feature_cache[cache_key] = features
                edge_features.append(features)
            
            # 3. 预计算相似度矩阵 | Precompute similarity matrices
            self.logger.info("计算相似度矩阵... | Calculating similarity matrices...")
            H, V = self.precompute_similarities(edge_features)
            
            # 4. 启发式搜索最优排列 | Heuristic search for optimal arrangement
            self.logger.info("搜索最优排列... | Searching for optimal arrangement...")
            grid = self.optimized_search(H, V)
            
            if grid is None:
                self.logger.warning("启发式搜索失败，使用全排列 | Heuristic search failed, using full permutation")
                grid = self.fallback_full_search(H, V)
                
            if grid is None:
                self.logger.error("无法找到有效排列，使用备用策略 | Cannot find valid arrangement, using fallback strategy")
                signal.alarm(0)
                return self.fallback_strategy()
            
            correct_sequence = grid[0] + grid[1] + grid[2]
            
            # 5. 生成交换步骤 | Generate swap sequence
            self.logger.info("生成交换序列... | Generating swap sequence...")
            initial_sequence = list(range(9))
            swap_sequence = self.generate_swap_sequence(initial_sequence, correct_sequence)
            
            elapsed = time.time() - start_time
            self.logger.info(f"验证码破解成功! 耗时: {elapsed:.2f}秒, 交换步骤: {len(swap_sequence)}次 | Captcha solved successfully! Time: {elapsed:.2f}s, Swap steps: {len(swap_sequence)}")
            
            # 保存结果图片 | Save result image
            self.save_result_grid(images, grid, folder="result_images")
            
            signal.alarm(0)
            return swap_sequence, grid
            
        except TimeoutException:
            self.logger.warning(f"验证码破解超时 ({self.timeout}秒)，使用备用策略 | Captcha solving timed out ({self.timeout}s), using fallback strategy")
            return self.fallback_strategy(), None
        except Exception as e:
            self.logger.error(f"验证码破解失败: {str(e)} | Captcha solving failed: {str(e)}")
            return self.fallback_strategy(), None
        finally:
            signal.alarm(0)
    
    def fallback_full_search(self, H, V):
        """
        备选方案：完整搜索
        Fallback plan: Complete search
        
        Args:
            H: 水平相似度矩阵 | Horizontal similarity matrix
            V: 垂直相似度矩阵 | Vertical similarity matrix
        
        Returns:
            list: 最优排列的网格 | Optimal arrangement grid
        """
        n = H.shape[0]
        indices = list(range(n))
        best_score = -float('inf')
        best_grid = None
        
        start_time = time.time()
        
        # 仅尝试部分组合以节省时间
        for group1 in itertools.combinations(indices, 3):
            rem = list(set(indices) - set(group1))
            for group2 in itertools.combinations(rem, 3):
                group3 = list(set(rem) - set(group2))
                
                # 每组内全排列
                for perm1 in itertools.permutations(group1):
                    for perm2 in itertools.permutations(group2):
                        for perm3 in itertools.permutations(group3):
                            grid = [list(perm1), list(perm2), list(perm3)]
                            score = self.evaluate_grid(grid, H, V)
                            if score > best_score:
                                best_score = score
                                best_grid = grid
                            if time.time() - start_time > 2.0:  # 最多搜索2秒
                                return best_grid
        return best_grid
    
    def fallback_strategy(self):
        """
        备用策略：随机交换几对图片
        Fallback strategy: Randomly swap a few pairs of images
        
        Returns:
            list: 随机交换步骤 | Random swap steps
        """
        self.logger.warning("使用备用策略 | Using fallback strategy")
        swaps = []
        positions = list(range(9))
        for _ in range(2):  # 交换两对
            idx1, idx2 = random.sample(positions, 2)
            swaps.append((idx1, idx2))
            positions.remove(idx1)
            positions.remove(idx2)
        return swaps
    
    def simulate_swaps(self, swap_sequence):
        """
        模拟交换过程（用于测试）
        Simulate swap process (for testing)
        
        Args:
            swap_sequence: 交换步骤列表 | List of swap steps
        
        Returns:
            list: 最终排列 | Final arrangement
        """
        arr = list(range(9))
        print("初始顺序 | Initial order:", arr)
        for i, (idx1, idx2) in enumerate(swap_sequence):
            arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
            print(f"交换 {i+1} | Swap {i+1}: 位置 | Position {idx1} 和 | and {idx2} -> {arr}")
        print("最终顺序 | Final order:", arr)
        return arr
    
    def save_images(self, images, folder="captcha_images"):
        """
        保存图片用于调试
        Save images for debugging
        
        Args:
            images: 图片列表 | List of images
            folder: 保存文件夹 | Save folder
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        for i, img in enumerate(images):
            img_path = os.path.join(folder, f"image_{i}.png")
            img.save(img_path)
            self.logger.info(f"保存图片 | Saved image: {img_path}")
    
    def save_result_grid(self, images, grid, folder="result_images"):
        """
        保存排列后的结果图片
        Save the arranged result image
        
        Args:
            images: 图片列表 | List of images
            grid: 网格排列 | Grid arrangement
            folder: 保存文件夹 | Save folder
        
        Returns:
            str: 结果图片路径 | Result image path
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # 创建空白大图 | Create empty large image
        img_width, img_height = images[0].size
        result_img = Image.new('RGB', (img_width * 3, img_height * 3))
        
        # 按正确顺序排列图片 | Arrange images in correct order
        for row_idx, row in enumerate(grid):
            for col_idx, img_idx in enumerate(row):
                img = images[img_idx]
                result_img.paste(img, (col_idx * img_width, row_idx * img_height))
        
        # 保存结果 | Save result
        result_path = os.path.join(folder, "result.jpg")
        result_img.save(result_path)
        self.logger.info(f"保存排列结果 | Saved arrangement result: {result_path}")
        
        return result_path
    
    def visualize_grid(self, images, grid, title="Correct Arrangement"):
        """
        可视化排列结果
        Visualize arrangement result
        
        Args:
            images: 图片列表 | List of images
            grid: 网格排列 | Grid arrangement
            title: 图表标题 | Chart title
        """
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(title, fontsize=16)
        
        for row_idx, row in enumerate(grid):
            for col_idx, img_idx in enumerate(row):
                ax = axes[row_idx, col_idx]
                ax.imshow(np.array(images[img_idx]))
                ax.set_title(f"Pos: ({row_idx},{col_idx})\nOrig: {img_idx}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig("visualization.png")
        plt.show()
    
    def test_local_images(self, folder_path):
        """
        使用本地图片测试验证码破解功能
        Test captcha solving functionality using local images
        
        Args:
            folder_path: 包含9张图片的文件夹路径 | Path to folder containing 9 images
        
        Returns:
            tuple: (交换步骤, 最终网格) | (Swap steps, final grid)
        """
        # 加载本地图片 | Load local images
        image_files = glob.glob(os.path.join(folder_path, "*.png"))
        image_files += glob.glob(os.path.join(folder_path, "*.jpg"))
        image_files += glob.glob(os.path.join(folder_path, "*.webp"))
        image_files = sorted(image_files)[:9]  # 只取前9张 | Only take first 9
        
        if len(image_files) < 9:
            self.logger.error(f"需要9张图片，只找到{len(image_files)}张 | Need 9 images, only found {len(image_files)}")
            return None, None
        
        images = []
        for file_path in image_files:
            try:
                img = Image.open(file_path)
                images.append(img)
                self.logger.info(f"加载图片 | Loaded image: {os.path.basename(file_path)}")
            except Exception as e:
                self.logger.error(f"无法加载图片 | Cannot load image {file_path}: {str(e)}")
        
        if len(images) != 9:
            self.logger.error(f"成功加载图片数 | Successfully loaded images: {len(images)}/9")
            return None, None
        
        # 保存原始图片 | Save original images
        self.save_images(images, folder="test_original")
        
        # 提取边缘特征 | Extract edge features
        self.logger.info("提取边缘特征... | Extracting edge features...")
        edge_features = [self.extract_edges(img) for img in images]
        
        # 预计算相似度矩阵 | Precompute similarity matrices
        self.logger.info("计算相似度矩阵... | Calculating similarity matrices...")
        H, V = self.precompute_similarities(edge_features)
        
        # 搜索最优排列 | Search for optimal arrangement
        self.logger.info("搜索最优排列... | Searching for optimal arrangement...")
        grid = self.optimized_search(H, V)
        
        if grid is None:
            self.logger.warning("启发式搜索失败，使用全排列 | Heuristic search failed, using full permutation")
            grid = self.fallback_full_search(H, V)
        
        if grid is None:
            self.logger.error("无法找到有效排列 | Cannot find valid arrangement")
            return None, None
        
        correct_sequence = grid[0] + grid[1] + grid[2]
        
        # 生成交换步骤 | Generate swap sequence
        self.logger.info("生成交换序列... | Generating swap sequence...")
        initial_sequence = list(range(9))
        swap_sequence = self.generate_swap_sequence(initial_sequence, correct_sequence)
        
        # 可视化结果 | Visualize result
        self.visualize_grid(images, grid, title="Solved Arrangement")
        
        # 保存结果图片 | Save result image
        result_path = self.save_result_grid(images, grid, folder="test_result")
        
        self.logger.info(f"测试完成! 交换步骤 | Test completed! Swap steps: {swap_sequence}")
        self.logger.info(f"最终排列 | Final arrangement: {grid}")
        self.logger.info(f"结果图片已保存至 | Result image saved to: {result_path}")
        
        return swap_sequence, grid
    
    def solve_from_file(self, filename, match_id=None):
        """
        从文件读取HTML内容并解决验证码
        Read HTML content from file and solve captcha
        
        Args:
            filename: HTML文件路径 | HTML file path
            match_id: 可选的匹配ID | Optional match ID
        
        Returns:
            tuple: (交换步骤, 最终网格) | (Swap steps, final grid)
        """
        with open(filename, "r", encoding="utf-8") as f:
            html_content = f.read()
        return self.solve_captcha(html_content, match_id)
    
    def solve_from_url(self, url, match_id=None):
        """
        从URL获取HTML内容并解决验证码
        Get HTML content from URL and solve captcha
        
        Args:
            url: 目标URL | Target URL
            match_id: 可选的匹配ID | Optional match ID
        
        Returns:
            tuple: (交换步骤, 最终网格) | (Swap steps, final grid)
        """
        response = requests.get(url)
        response.raise_for_status()
        return self.solve_captcha(response.text, match_id) 