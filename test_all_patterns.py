import copy

def _create_north_pattern():
    """北向き時計回りパターン（左列→上→右列）"""
    return [
        {"name": "stair1", "x": 0, "y": 0, "z": 0.193, "x_len": 0.5, "y_len": 0.125},
        {"name": "stair7", "x": 0, "y": 0.833, "z": 1.351, "x_len": 0.5, "y_len": 0.167},
        {"name": "stair14", "x": 0.5, "y": 0, "z": 2.702, "x_len": 0.5, "y_len": 0.125},
    ]

def _create_south_pattern():
    """南向き時計回りパターン（右列→下→左列）"""
    return [
        {"name": "stair1", "x": 0.5, "y": 0.875, "z": 0.193, "x_len": 0.5, "y_len": 0.125},
        {"name": "stair7", "x": 0.5, "y": 0, "z": 1.351, "x_len": 0.5, "y_len": 0.167},
        {"name": "stair14", "x": 0, "y": 0.875, "z": 2.702, "x_len": 0.5, "y_len": 0.125},
    ]

def _create_east_pattern():
    """東向き時計回りパターン（下列→右→上列）"""
    return [
        {"name": "stair1", "x": 0, "y": 0, "z": 0.193, "x_len": 0.125, "y_len": 0.5},
        {"name": "stair7", "x": 0.833, "y": 0, "z": 1.351, "x_len": 0.167, "y_len": 0.5},
        {"name": "stair14", "x": 0, "y": 0.5, "z": 2.702, "x_len": 0.125, "y_len": 0.5},
    ]

def _create_west_pattern():
    """西向き時計回りパターン（上列→左→下列）"""
    return [
        {"name": "stair1", "x": 0.875, "y": 0.5, "z": 0.193, "x_len": 0.125, "y_len": 0.5},
        {"name": "stair7", "x": 0, "y": 0.5, "z": 1.351, "x_len": 0.167, "y_len": 0.5},
        {"name": "stair14", "x": 0.875, "y": 0, "z": 2.702, "x_len": 0.125, "y_len": 0.5},
    ]

def _mirror_x_for_north_south(steps):
    """南北方向: X軸鏡像（時計回り→反時計回り）"""
    mirrored = []
    for step in steps:
        s = copy.deepcopy(step)
        right_edge = s['x'] + s['x_len']
        s['x'] = 1.0 - right_edge
        mirrored.append(s)
    return mirrored

def _mirror_y_for_east_west(steps):
    """東西方向: Y軸鏡像（時計回り→反時計回り）"""
    mirrored = []
    for step in steps:
        s = copy.deepcopy(step)
        top_edge = s['y'] + s['y_len']
        s['y'] = 1.0 - top_edge
        mirrored.append(s)
    return mirrored

def print_pattern(name, steps):
    print(f"\n=== {name} ===")
    for s in steps:
        x, y = s['x'], s['y']
        x_len, y_len = s['x_len'], s['y_len']
        print(f"{s['name']}: ({x:.3f}, {y:.3f}) - ({x+x_len:.3f}, {y+y_len:.3f})")
    
    # 範囲チェック
    all_ok = True
    for s in steps:
        x, y = s['x'], s['y']
        x_len, y_len = s['x_len'], s['y_len']
        if x < 0 or x + x_len > 1.0 or y < 0 or y + y_len > 1.0:
            print(f"  ⚠️ {s['name']}: 範囲外！")
            all_ok = False
    if all_ok:
        print("  ✓ 全て0-1の範囲内")

# 北向き
north_cw = _create_north_pattern()
north_ccw = _mirror_x_for_north_south(north_cw)
print_pattern("北↑ 時計回り", north_cw)
print_pattern("北↑ 反時計回り", north_ccw)

# 南向き
south_cw = _create_south_pattern()
south_ccw = _mirror_x_for_north_south(south_cw)
print_pattern("南↓ 時計回り", south_cw)
print_pattern("南↓ 反時計回り", south_ccw)

# 東向き
east_cw = _create_east_pattern()
east_ccw = _mirror_y_for_east_west(east_cw)
print_pattern("東→ 時計回り", east_cw)
print_pattern("東→ 反時計回り", east_ccw)

# 西向き
west_cw = _create_west_pattern()
west_ccw = _mirror_y_for_east_west(west_cw)
print_pattern("西← 時計回り", west_cw)
print_pattern("西← 反時計回り", west_ccw)

print("\n=== 検証結果 ===")
print("北↑ 時計回り stair7: 左上に配置 (0, 0.833)-(0.5, 1.0)")
print("北↑ 反時計回り stair7: 右上に配置 (0.5, 0.833)-(1.0, 1.0)")
print("南↓ 時計回り stair7: 右下に配置 (0.5, 0)-(1.0, 0.167)")
print("南↓ 反時計回り stair7: 左下に配置 (0, 0)-(0.5, 0.167)")
print("東→ 時計回り stair7: 右下に配置 (0.833, 0)-(1.0, 0.5)")
print("東→ 反時計回り stair7: 右上に配置 (0.833, 0.5)-(1.0, 1.0)")
print("西← 時計回り stair7: 左上に配置 (0, 0.5)-(0.167, 1.0)")
print("西← 反時計回り stair7: 左下に配置 (0, 0)-(0.167, 0.5)")
