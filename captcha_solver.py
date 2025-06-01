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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CaptchaSolver')

# 全局缓存
feature_cache = {}
image_cache = {}

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Captcha solving timed out")

def download_image(url):
    """下载单个图片并缓存"""
    if url in image_cache:
        return image_cache[url]
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        image_cache[url] = img
        return img
    except Exception as e:
        logger.error(f"下载图片失败: {url} - {str(e)}")
        return None

def download_images(html_content, match_id=None):
    """从HTML内容提取图片URL并并行下载图片"""
    soup = BeautifulSoup(html_content, 'html.parser')
    img_tags = soup.select('.grid-item img')
    
    if not img_tags:
        logger.error("未找到图片元素")
        return []
    
    urls = []
    for img in img_tags:
        src = img.get('src', '')
        if not src:
            continue
            
        if src.startswith('//'):
            url = 'https:' + src
        elif src.startswith('/'):
            url = 'https://example.com' + src
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
        future_to_url = {executor.submit(download_image, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                img = future.result()
                if img:
                    images.append(img)
            except Exception as e:
                logger.error(f"图片下载失败: {url} - {str(e)}")
    
    return images

def extract_edges(img, edge_size=20):
    """提取图片四边特征并压缩"""
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

def precompute_similarities(edge_features):
    """预计算相似度矩阵"""
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

def optimized_search(H, V):
    """使用贪心+回溯的启发式搜索最优排列"""
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
        score = evaluate_grid(grid, H, V)
        if score > best_score:
            best_score = score
            best_grid = [row[:] for row in grid]
    
    return best_grid

def evaluate_grid(grid, H, V):
    """评估网格得分"""
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

def generate_swap_sequence(initial, correct):
    """生成交换步骤"""
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

def solve_captcha(html_content, match_id=None, timeout=5):
    """主函数：解决验证码"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        start_time = time.time()
        
        # 1. 下载图片
        logger.info("开始下载图片...")
        images = download_images(html_content, match_id)
        
        if len(images) != 9:
            logger.error(f"图片数量错误: 需要9张, 实际获取{len(images)}张")
            signal.alarm(0)
            return fallback_strategy()
        
        # 保存原始图片用于调试
        save_images(images, folder="original_images")
        
        # 2. 提取边缘特征
        logger.info("提取边缘特征...")
        edge_features = []
        for i, img in enumerate(images):
            cache_key = f"{match_id}_{i}" if match_id else None
            if cache_key and cache_key in feature_cache:
                features = feature_cache[cache_key]
            else:
                features = extract_edges(img)
                if cache_key:
                    feature_cache[cache_key] = features
            edge_features.append(features)
        
        # 3. 预计算相似度矩阵
        logger.info("计算相似度矩阵...")
        H, V = precompute_similarities(edge_features)
        
        # 4. 启发式搜索最优排列
        logger.info("搜索最优排列...")
        grid = optimized_search(H, V)
        
        if grid is None:
            logger.warning("启发式搜索失败，使用全排列")
            grid = fallback_full_search(H, V)
            
        if grid is None:
            logger.error("无法找到有效排列，使用备用策略")
            signal.alarm(0)
            return fallback_strategy()
        
        correct_sequence = grid[0] + grid[1] + grid[2]
        
        # 5. 生成交换步骤
        logger.info("生成交换序列...")
        initial_sequence = list(range(9))
        swap_sequence = generate_swap_sequence(initial_sequence, correct_sequence)
        
        elapsed = time.time() - start_time
        logger.info(f"验证码破解成功! 耗时: {elapsed:.2f}秒, 交换步骤: {len(swap_sequence)}次")
        
        # 保存结果图片
        save_result_grid(images, grid, folder="result_images")
        
        signal.alarm(0)
        return swap_sequence, grid
        
    except TimeoutException:
        logger.warning(f"验证码破解超时 ({timeout}秒)，使用备用策略")
        return fallback_strategy(), None
    except Exception as e:
        logger.error(f"验证码破解失败: {str(e)}")
        return fallback_strategy(), None
    finally:
        signal.alarm(0)

def fallback_full_search(H, V):
    """备选方案：完整搜索"""
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
                        score = evaluate_grid(grid, H, V)
                        if score > best_score:
                            best_score = score
                            best_grid = grid
                        if time.time() - start_time > 2.0:  # 最多搜索2秒
                            return best_grid
    return best_grid

def fallback_strategy():
    """备用策略：随机交换几对图片"""
    logger.warning("使用备用策略")
    swaps = []
    positions = list(range(9))
    for _ in range(2):  # 交换两对
        idx1, idx2 = random.sample(positions, 2)
        swaps.append((idx1, idx2))
        positions.remove(idx1)
        positions.remove(idx2)
    return swaps

def simulate_swaps(swap_sequence):
    """模拟交换过程（用于测试）"""
    arr = list(range(9))
    print("初始顺序:", arr)
    for i, (idx1, idx2) in enumerate(swap_sequence):
        arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
        print(f"交换 {i+1}: 位置 {idx1} 和 {idx2} -> {arr}")
    print("最终顺序:", arr)
    return arr

def save_images(images, folder="captcha_images"):
    """保存图片用于调试"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for i, img in enumerate(images):
        img_path = os.path.join(folder, f"image_{i}.png")
        img.save(img_path)
        logger.info(f"保存图片: {img_path}")

def save_result_grid(images, grid, folder="result_images"):
    """保存排列后的结果图片"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 创建空白大图
    img_width, img_height = images[0].size
    result_img = Image.new('RGB', (img_width * 3, img_height * 3))
    
    # 按正确顺序排列图片
    for row_idx, row in enumerate(grid):
        for col_idx, img_idx in enumerate(row):
            img = images[img_idx]
            result_img.paste(img, (col_idx * img_width, row_idx * img_height))
    
    # 保存结果
    result_path = os.path.join(folder, "result.jpg")
    result_img.save(result_path)
    logger.info(f"保存排列结果: {result_path}")
    
    return result_path

def visualize_grid(images, grid, title="Correct Arrangement"):
    """可视化排列结果"""
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

def test_local_images(folder_path):
    """
    使用本地图片测试验证码破解功能
    :param folder_path: 包含9张图片的文件夹路径
    :return: 交换步骤和最终网格
    """
    # 加载本地图片
    image_files = glob.glob(os.path.join(folder_path, "*.png"))
    image_files += glob.glob(os.path.join(folder_path, "*.jpg"))
    image_files = glob.glob(os.path.join(folder_path, "*.webp"))
    image_files = sorted(image_files)[:9]  # 只取前9张
    
    if len(image_files) < 9:
        logger.error(f"需要9张图片，只找到{len(image_files)}张")
        return None, None
    
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            images.append(img)
            logger.info(f"加载图片: {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"无法加载图片 {file_path}: {str(e)}")
    
    if len(images) != 9:
        logger.error(f"成功加载图片数: {len(images)}/9")
        return None, None
    
    # 保存原始图片
    save_images(images, folder="test_original")
    
    # 提取边缘特征
    logger.info("提取边缘特征...")
    edge_features = [extract_edges(img) for img in images]
    
    # 预计算相似度矩阵
    logger.info("计算相似度矩阵...")
    H, V = precompute_similarities(edge_features)
    
    # 搜索最优排列
    logger.info("搜索最优排列...")
    grid = optimized_search(H, V)
    
    if grid is None:
        logger.warning("启发式搜索失败，使用全排列")
        grid = fallback_full_search(H, V)
    
    if grid is None:
        logger.error("无法找到有效排列")
        return None, None
    
    correct_sequence = grid[0] + grid[1] + grid[2]
    
    # 生成交换步骤
    logger.info("生成交换序列...")
    initial_sequence = list(range(9))
    swap_sequence = generate_swap_sequence(initial_sequence, correct_sequence)
    
    # 可视化结果
    visualize_grid(images, grid, title="Solved Arrangement")
    
    # 保存结果图片
    result_path = save_result_grid(images, grid, folder="test_result")
    
    logger.info(f"测试完成! 交换步骤: {swap_sequence}")
    logger.info(f"最终排列: {grid}")
    logger.info(f"结果图片已保存至: {result_path}")
    
    return swap_sequence, grid

def solve_from_file(filename, match_id=None):
    """从文件读取HTML内容并解决验证码"""
    with open(filename, "r", encoding="utf-8") as f:
        html_content = f.read()
    return solve_captcha(html_content, match_id)

def solve_from_url(url, match_id=None):
    """从URL获取HTML内容并解决验证码"""
    response = requests.get(url)
    response.raise_for_status()
    return solve_captcha(response.text, match_id)

# 使用示例
if __name__ == "__main__":
    print("九宫格验证码破解系统")
    print("=" * 50)
    print("1. 从本地图片测试")
    print("2. 从HTML文件解决")
    print("3. 从URL解决")
    choice = input("请选择模式 (1/2/3): ")
    
    if choice == "1":
        folder = input("请输入包含9张图片的文件夹路径: ")
        swaps, grid = test_local_images(folder)
        if swaps:
            print("\n交换步骤:", swaps)
            print("最终网格排列:")
            for i, row in enumerate(grid):
                print(f"行 {i+1}: {row}")
            
            # 模拟交换过程
            print("\n模拟交换过程:")
            simulate_swaps(swaps)
    
    elif choice == "2":
        filename = input("请输入HTML文件路径: ")
        match_id = input("请输入match_id (可选): ") or None
        swaps, grid = solve_from_file(filename, match_id)
        if swaps:
            print("\n交换步骤:", swaps)
            print("最终网格排列:")
            for i, row in enumerate(grid):
                print(f"行 {i+1}: {row}")
            
            # 模拟交换过程
            print("\n模拟交换过程:")
            simulate_swaps(swaps)
    
    elif choice == "3":
        url = input("请输入URL: ")
        match_id = input("请输入match_id (可选): ") or None
        swaps, grid = solve_from_url(url, match_id)
        if swaps:
            print("\n交换步骤:", swaps)
            print("最终网格排列:")
            for i, row in enumerate(grid):
                print(f"行 {i+1}: {row}")
            
            # 模拟交换过程
            print("\n模拟交换过程:")
            simulate_swaps(swaps)
    
    else:
        print("无效选择!")
