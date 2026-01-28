#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D座標データを2D平面図として可視化
"""
import json
import numpy as np
import cv2
import sys

def visualize_3d_walls(json_path, output_image="walls_3d_visualization.png", scale=50, highlight_wall_ids=None, show_info_text=False, wall_color=(200, 200, 200), bg_color=(255, 255, 255)):
    """
    3D座標JSONを読み込んで2D平面図として描画
    
    Parameters:
    - scale: メートルからピクセルへの変換スケール（大きいほど拡大）
    - highlight_wall_ids: 赤色で強調表示する壁のIDリスト
    - show_info_text: スケール情報を表示するか
    - wall_color: 壁の色 (BGR) デフォルト: ライトグレー(200,200,200)
    - bg_color: 背景色 (BGR) デフォルト: 白(255,255,255)
    """
    print(f"Loading: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    walls = data['walls']
    print(f"Total walls: {len(walls)}")
    
    # すべての座標から画像サイズを決定
    all_x = []
    all_y = []
    
    for wall in walls:
        all_x.extend([wall['start'][0], wall['end'][0]])
        all_y.extend([wall['start'][1], wall['end'][1]])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # 画像サイズを計算（余白込み）
    margin = 50
    img_width = int((max_x - min_x) * scale) + 2 * margin
    img_height = int((max_y - min_y) * scale) + 2 * margin
    
    print(f"Image size: {img_width} x {img_height}")
    print(f"Floor plan bounds: {min_x:.2f} to {max_x:.2f} m (X), {min_y:.2f} to {max_y:.2f} m (Y)")
    
    # 白背景の画像を作成
    canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    
    # 壁を描画
    for wall in walls:
        # メートル → ピクセル座標
        x1 = int((wall['start'][0] - min_x) * scale) + margin
        y1 = int((wall['start'][1] - min_y) * scale) + margin
        x2 = int((wall['end'][0] - min_x) * scale) + margin
        y2 = int((wall['end'][1] - min_y) * scale) + margin
        
        # Y軸反転（画像座標系）
        y1 = img_height - y1
        y2 = img_height - y2
        
        # 壁の太さを計算
        thickness_px = max(2, int(wall['thickness'] * scale))
        
        # 強調表示する壁は赤色、それ以外は指定の壁色
        wall_id = wall.get('id')
        if highlight_wall_ids and wall_id in highlight_wall_ids:
            color = (0, 0, 255)  # 赤色 (BGR)
            thickness_px = max(3, thickness_px + 1)  # 少し太く
        else:
            color = wall_color
        
        # 線を描画
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness_px)
    
    # グリッド線を描画（0.45mごと = 45cm）
    grid_color = (200, 200, 200)
    grid_spacing = 0.45  # visualizationの1マス = 45cm（一条工務店の図面の1マス相当）
    for x_m in np.arange(0, max_x - min_x + 1, grid_spacing):
        x_px = int(x_m * scale) + margin
        cv2.line(canvas, (x_px, 0), (x_px, img_height), grid_color, 1)
    
    for y_m in np.arange(0, max_y - min_y + 1, grid_spacing):
        y_px = int(y_m * scale) + margin
        y_px_inv = img_height - y_px
        cv2.line(canvas, (0, y_px_inv), (img_width, y_px_inv), grid_color, 1)
    
    # スケール・総数などの説明テキスト（任意表示）
    if show_info_text:
        furniture_count = len(data.get('furniture', []))
        scale_text = f"Scale: 1m = {scale}px | grid: 45cm | Total: {len(walls)} walls"
        if furniture_count > 0:
            scale_text += f" | Furniture: {furniture_count}"
        cv2.putText(canvas, scale_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 家具オブジェクトを描画
    if 'furniture' in data and len(data['furniture']) > 0:
        for furniture in data['furniture']:
            bounds = furniture['bounds']
            x_start_m = bounds['x_start']
            y_start_m = bounds['y_start']
            x_end_m = bounds['x_end']
            y_end_m = bounds['y_end']
            
            # メートル → ピクセル座標に変換
            x1_px = int((x_start_m - min_x) * scale) + margin
            y1_px = int((y_start_m - min_y) * scale) + margin
            x2_px = int((x_end_m - min_x) * scale) + margin
            y2_px = int((y_end_m - min_y) * scale) + margin
            
            # Y軸反転
            y1_px = img_height - y1_px
            y2_px = img_height - y2_px
            
            # 家具の色（JSONから取得、デフォルトは茶色）
            if 'color_rgb' in furniture:
                furniture_color = tuple(furniture['color_rgb'])
            else:
                furniture_color = (19, 69, 139)  # BGR: デフォルト茶色
            
            # 矩形として描画（塗りつぶし）
            cv2.rectangle(canvas, (x1_px, y2_px), (x2_px, y1_px), furniture_color, -1)
            
            # 輪郭線を描画（少し濃い色）
            outline_color = (10, 40, 80)
            cv2.rectangle(canvas, (x1_px, y2_px), (x2_px, y1_px), outline_color, 2)
    
    # 床を描画
    if 'floors' in data and len(data['floors']) > 0:
        for floor in data['floors']:
            x1_m = floor['x1']
            y1_m = floor['y1']
            x2_m = floor['x2']
            y2_m = floor['y2']
            
            # メートル → ピクセル座標に変換
            x1_px = int((x1_m - min_x) * scale) + margin
            y1_px = int((y1_m - min_y) * scale) + margin
            x2_px = int((x2_m - min_x) * scale) + margin
            y2_px = int((y2_m - min_y) * scale) + margin
            
            # Y軸反転
            y1_px = img_height - y1_px
            y2_px = img_height - y2_px
            
            # 床の色（薄い緑色）
            floor_color = (200, 255, 200)  # BGR: 薄い緑
            
            # 矩形として描画（半透明の塗りつぶし）
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1_px, y2_px), (x2_px, y1_px), floor_color, -1)
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
            
            # 輪郭線を描画（緑色）
            outline_color = (0, 150, 0)
            cv2.rectangle(canvas, (x1_px, y2_px), (x2_px, y1_px), outline_color, 2)
    
    # 保存
    cv2.imwrite(output_image, canvas)
    print(f"Saved: {output_image}")
    
    return canvas

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_3d_walls.py <JSON_PATH> [output.png] [scale]")
        print()
        print("Example:")
        print("  python visualize_3d_walls.py walls_3d.json walls_viz.png 50")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else "walls_3d_visualization.png"
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    visualize_3d_walls(json_path, output_image, scale, show_info_text=True)
