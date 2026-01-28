#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画像ファイルから直接壁線を抽出するスクリプト
（PDFを経由せずにJPG/PNGから処理）

使用方法:
  python refine_from_image.py input.jpg 180 12
  python refine_from_image.py input.png 200 8
"""

import sys
import cv2
import numpy as np
from skimage import measure
from pathlib import Path

def extract_thick_lines(image, black_threshold=180, min_thickness=10):
    """
    画像から黒線を抽出（改良版：細い線も保持）
    
    Parameters:
    - image: グレースケール画像
    - black_threshold: 黒として扱う閾値（この値以下を黒とみなす）
    - min_thickness: 最小線幅（ピクセル） ※モルフォロジー演算のカーネルサイズに使用
    """
    # 2値化（黒線を白に反転）
    _, binary = cv2.threshold(image, black_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # min_thicknessが小さい場合は膨張処理で線を太くしてから細線除去
    # これにより、細い線を保持しつつノイズだけを除去
    if min_thickness <= 6:
        # 細い線設定：軽い膨張→収縮でノイズ除去（線は保持）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # クロージング（膨張→収縮）で小さな隙間を埋める
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        # オープニング（収縮→膨張）で小さなノイズを除去
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        # 太い線設定：距離変換を使用（従来の動作）
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        radius_threshold = max(1.0, min_thickness * 0.35)
        _, processed = cv2.threshold(dist_transform, radius_threshold, 255, cv2.THRESH_BINARY)
        processed = processed.astype(np.uint8)
    
    print(f"  線幅フィルタ: min_thickness={min_thickness} ({'モルフォロジー' if min_thickness <= 6 else f'距離変換 r={max(1.0, min_thickness * 0.35):.2f}'})")
    
    return processed

def remove_small_noise(binary_image, min_area=100):
    """小さなノイズを除去"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, 8, cv2.CV_32S
    )
    
    output = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):  # 0はバックグラウンド
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_area:
            output[labels == i] = 255
    
    return output

def remove_isolated_shapes(binary_image, circularity_threshold=0.3, min_thickness=8):
    """
    孤立した図形を除去（ロゴや記号など）
    
    Parameters:
    - binary_image: 2値画像
    - circularity_threshold: 円形度の閾値（これ以上なら円形とみなす）
    - min_thickness: 最小線幅（細い線設定の場合は除去条件を緩和）
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = binary_image.copy()
    
    # min_thicknessが小さい場合は、より慎重に除去（壁線を保護）
    # 細い線設定では、面積閾値を小さくし、アスペクト比条件を厳しくする
    if min_thickness <= 8:
        area_threshold = 1000  # 小さな図形のみ除去
        aspect_ratio_threshold = 10  # より細長いもののみ除去
        max_area_to_remove = 10000  # この面積以上の大きな構造は保護
    else:
        area_threshold = 5000
        aspect_ratio_threshold = 5
        max_area_to_remove = 50000
    
    removed_count = 0
    total_count = len(contours)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        
        # 大きな構造（間取り全体など）は保護
        if area > max_area_to_remove:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        # 円形度: 4π×面積 / 周囲長^2
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 外接矩形を取得してアスペクト比を計算
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # 除去条件:
        # 1. 円形に近い（ロゴなど）
        # 2. 細長い（矢印など）: アスペクト比>threshold かつ 面積<threshold
        should_remove = False
        if circularity > circularity_threshold:
            should_remove = True
        elif aspect_ratio > aspect_ratio_threshold and area < area_threshold:
            should_remove = True
        
        if should_remove:
            cv2.drawContours(output, [contour], -1, 0, -1)
            removed_count += 1
    
    print(f"    除去した図形: {removed_count}/{total_count} (最大保護面積: {max_area_to_remove})")
    
    return output

def remove_corner_logos(binary_image, corner_margin_ratio=0.10):
    """
    四隅のロゴや記号を除去（改良版：壁を保護）
    
    Parameters:
    - binary_image: 2値画像
    - corner_margin_ratio: 隅とみなす領域の比率（デフォルト10%）
    
    Returns:
    - ロゴ除去後の画像
    """
    h, w = binary_image.shape
    margin_h = int(h * corner_margin_ratio)
    margin_w = int(w * corner_margin_ratio)
    
    output = binary_image.copy()
    
    # 四隅の領域を定義（右下のみ処理 - ロゴは通常右下にある）
    corners = [
        ("右下", h - margin_h, h, w - margin_w, w),
    ]
    
    removed_count = 0
    
    for corner_name, y1, y2, x1, x2 in corners:
        corner_region = binary_image[y1:y2, x1:x2]
        white_pixels = np.sum(corner_region == 255)
        corner_area = (y2 - y1) * (x2 - x1)
        
        # 隅領域の白ピクセル密度を計算
        density = white_pixels / corner_area
        
        # ロゴの特徴：
        # 1. 適度なサイズ（大きすぎず小さすぎず）
        # 2. 密度が低い（壁は高密度、ロゴは低密度）
        # 3. コンパクト（孤立した成分）
        
        if 1000 < white_pixels < 50000 and density < 0.3:
            # 連結成分を解析
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                corner_region.astype(np.uint8), 8
            )
            
            # 小さな孤立成分の数が多い場合、ロゴの可能性が高い
            small_components = sum(1 for i in range(1, num_labels) 
                                  if stats[i, cv2.CC_STAT_AREA] < 5000)
            
            if small_components > 2 or density < 0.1:
                # ロゴと判断して除去
                output[y1:y2, x1:x2] = 0
                removed_count += 1
                print(f"  {corner_name}の要素を除去（{white_pixels}ピクセル, 密度={density:.2%}）")
            else:
                print(f"  {corner_name}は壁の可能性が高いため保持（密度={density:.2%}）")
        elif white_pixels > 50000:
            print(f"  {corner_name}は壁として保持（大きすぎる: {white_pixels}ピクセル）")
    
    if removed_count > 0:
        print(f"  合計{removed_count}箇所の隅要素を除去しました")
    else:
        print(f"  四隅に除去すべきロゴは検出されませんでした")
    
    return output

def find_main_floor_plan_region(binary_image):
    """
    間取り図の主要領域を検出（改良版）
    - 大きな成分はすべて保持
    - 小さなノイズのみ除去
    """
    # 連結成分を検出
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    if num_labels <= 1:
        return binary_image
    
    # 面積でソート（背景を除く）
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    
    # 閾値を計算：最大成分の5%以上のサイズは保持
    if len(areas) > 0:
        max_area = areas[0][1]
        area_threshold = max(500, max_area * 0.05)  # 最小500ピクセルまたは最大の5%
    else:
        area_threshold = 500
    
    # 閾値以上の成分をすべて保持
    output = np.zeros_like(binary_image)
    kept_count = 0
    
    for label_id, area in areas:
        if area >= area_threshold:
            output[labels == label_id] = 255
            kept_count += 1
    
    print(f"  保持した成分: {kept_count}/{len(areas)}個")
    
    return output

def crop_to_content(binary_image, margin=20):
    """内容に合わせてクロップ"""
    coords = cv2.findNonZero(binary_image)
    if coords is None:
        return binary_image
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # マージンを追加
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(binary_image.shape[1] - x, w + 2 * margin)
    h = min(binary_image.shape[0] - y, h + 2 * margin)
    
    return binary_image[y:y+h, x:x+w]

def refine_floor_plan_from_image(
    image_path,
    black_threshold=180,
    min_thickness=12,
    remove_corners=False,
    margin_vertical=10,
    margin_horizontal=5
):
    """
    画像ファイルから壁線を抽出・精製
    
    Parameters:
    - image_path: 入力画像ファイルパス
    - black_threshold: 黒線の閾値
    - min_thickness: 最小線幅
    - remove_corners: 四隅のロゴを除去するか（デフォルトFalse）
    - margin_vertical: 上下除外率（％）
    - margin_horizontal: 左右除外率（％）
    
    Returns:
    - refined: 精製された画像
    - output_path: 出力ファイルパス
    """
    print(f"画像を読み込み中: {image_path}")
    
    # 画像読み込み（グレースケール）
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")
    
    print(f"画像サイズ: {image.shape[1]}x{image.shape[0]} ピクセル")
    
    # 除外範囲のクロップ処理
    if margin_vertical > 0 or margin_horizontal > 0:
        h, w = image.shape
        y1 = int(h * margin_vertical / 100)
        y2 = int(h * (100 - margin_vertical) / 100)
        x1 = int(w * margin_horizontal / 100)
        x2 = int(w * (100 - margin_horizontal) / 100)
        image = image[y1:y2, x1:x2]
        print(f"除外後のサイズ: {image.shape[1]}x{image.shape[0]} ピクセル (上下{margin_vertical}%, 左右{margin_horizontal}%除外)")
    
    # ステップ1: 太い黒線抽出
    print(f"太い黒線を抽出中（閾値={black_threshold}, 太さ>={min_thickness}px）...")
    thick_lines = extract_thick_lines(image, black_threshold, min_thickness)
    white_pixels_1 = np.sum(thick_lines > 0)
    print(f"  → 抽出後の白ピクセル数: {white_pixels_1}")
    
    # ステップ2: 小さなノイズ除去
    print("小さなノイズを除去中...")
    no_noise = remove_small_noise(thick_lines, min_area=100)
    white_pixels_2 = np.sum(no_noise > 0)
    print(f"  → ノイズ除去後の白ピクセル数: {white_pixels_2}")
    
    # ステップ3: 孤立した図形除去
    print("孤立した図形（ロゴなど）を除去中...")
    no_isolated = remove_isolated_shapes(no_noise, circularity_threshold=0.3, min_thickness=min_thickness)
    white_pixels_3 = np.sum(no_isolated > 0)
    print(f"  → 孤立図形除去後の白ピクセル数: {white_pixels_3}")
    
    # ステップ4: 四隅のロゴ除去（オプション）
    if remove_corners:
        print("四隅のロゴを除去中...")
        no_corners = remove_corner_logos(no_isolated, corner_margin_ratio=0.10)
        white_pixels_4 = np.sum(no_corners > 0)
        print(f"  → 四隅除去後の白ピクセル数: {white_pixels_4}")
    else:
        print("四隅のロゴ除去をスキップ（必要に応じて --roi または --remove-corners を使用）")
        no_corners = no_isolated
    
    # ステップ5: メイン領域抽出
    print("メイン間取り領域を抽出中...")
    main_region = find_main_floor_plan_region(no_corners)
    white_pixels_5 = np.sum(main_region > 0)
    print(f"  → メイン領域抽出後の白ピクセル数: {white_pixels_5}")
    
    if white_pixels_5 == 0:
        print("⚠️ 警告: すべての線が除去されました。パラメータを調整してください。")
        # 元の画像を返す（真っ白を避ける）
        return no_corners, f"refined_t{black_threshold}_w{min_thickness}.png"
    
    # ステップ6: クロップ
    print("内容に合わせてクロップ中...")
    cropped = crop_to_content(main_region, margin=20)
    
    # 白黒反転（黒線が黒になるように）
    refined = cv2.bitwise_not(cropped)
    
    # 保存
    output_path = f"refined_t{black_threshold}_w{min_thickness}.png"
    cv2.imwrite(output_path, refined)
    
    print(f"\n[OK] 完了: {output_path}")
    print(f"  出力サイズ: {refined.shape[1]}x{refined.shape[0]} ピクセル")
    
    return refined, output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python refine_from_image.py <画像ファイル> [黒閾値] [最小太さ]")
        print("\n例:")
        print("  python refine_from_image.py floor_plan.jpg 180 12")
        print("  python refine_from_image.py floor_plan.png 200 8")
        sys.exit(1)
    
    image_path = sys.argv[1]
    black_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    min_thickness = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    
    try:
        refine_floor_plan_from_image(image_path, black_threshold, min_thickness)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)
