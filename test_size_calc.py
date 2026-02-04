"""サイズ計算の検証"""

# 矩形サイズ例: 6m x 4m
rect_width_m = 6.0
rect_height_m = 4.0

print(f"矩形サイズ: {rect_width_m}m x {rect_height_m}m")
print()

# 北向き階段（南北方向）
print("=== 北向き時計回り ===")
print("パターン定義: x_len=0.5, y_len=0.125 or 0.167")
print()

# stair1: narrow (x_len=0.5, y_len=0.125)
width_m = rect_width_m * 0.5  # 3.0m
depth_m = rect_height_m * 0.125  # 0.5m
print(f"stair1 (narrow): width={width_m:.1f}m, depth={depth_m:.1f}m")
print(f"  → 矩形の左半分、高さは1/8")

# stair7: wide (x_len=0.5, y_len=0.167)
width_m = rect_width_m * 0.5  # 3.0m
depth_m = rect_height_m * 0.167  # 0.67m
print(f"stair7 (wide): width={width_m:.1f}m, depth={depth_m:.2f}m")
print(f"  → 矩形の左半分、高さは1/6")
print()

# 東向き階段（東西方向）
print("=== 東向き時計回り ===")
print("パターン定義: x_len=0.125 or 0.167, y_len=0.5")
print()

# stair1: narrow (x_len=0.125, y_len=0.5)
width_m = rect_width_m * 0.125  # 0.75m
depth_m = rect_height_m * 0.5  # 2.0m
print(f"stair1 (narrow): width={width_m:.2f}m, depth={depth_m:.1f}m")
print(f"  → 幅は矩形の1/8、高さは半分")

# stair7: wide (x_len=0.167, y_len=0.5)
width_m = rect_width_m * 0.167  # 1.0m
depth_m = rect_height_m * 0.5  # 2.0m
print(f"stair7 (wide): width={width_m:.1f}m, depth={depth_m:.1f}m")
print(f"  → 幅は矩形の1/6、高さは半分")
print()

print("="*60)
print("重要な確認:")
print("1. 南北方向: width（X方向）が長軸、depth（Y方向）が短軸")
print("2. 東西方向: width（X方向）が短軸、depth（Y方向）が長軸")
print("3. パターン定義のx_len, y_lenを直接使用して正しいサイズが得られる ✓")
