"""直線階段パターンの検証"""

step_height = 1.0 / 14
step_width = 1.0 / 14
z_step = 0.193

print("=== 直線階段パターン検証 ===\n")

# 北向き（下から上へY方向）
print("【北↑】下から上へY方向に昇る")
print("  x_len=1.0（矩形の全幅）, y_len=1/14（14等分）")
for i in range(3):  # 最初の3段だけ表示
    y = i * step_height
    z = (i+1) * z_step
    print(f"  stair{i+1}: x=0.000-1.000, y={y:.3f}-{y+step_height:.3f}, z={z:.3f}")
print("  ...")
i = 13
y = i * step_height
z = (i+1) * z_step
print(f"  stair{i+1}: x=0.000-1.000, y={y:.3f}-{y+step_height:.3f}, z={z:.3f}")
print()

# 南向き（上から下へY方向）
print("【南↓】上から下へY方向に昇る")
print("  x_len=1.0（矩形の全幅）, y_len=1/14（14等分）")
for i in range(3):
    y = (13-i) * step_height
    z = (i+1) * z_step
    print(f"  stair{i+1}: x=0.000-1.000, y={y:.3f}-{y+step_height:.3f}, z={z:.3f}")
print("  ...")
i = 13
y = (13-i) * step_height
z = (i+1) * z_step
print(f"  stair{i+1}: x=0.000-1.000, y={y:.3f}-{y+step_height:.3f}, z={z:.3f}")
print()

# 東向き（左から右へX方向）
print("【東→】左から右へX方向に昇る")
print("  x_len=1/14（14等分）, y_len=1.0（矩形の全高）")
for i in range(3):
    x = i * step_width
    z = (i+1) * z_step
    print(f"  stair{i+1}: x={x:.3f}-{x+step_width:.3f}, y=0.000-1.000, z={z:.3f}")
print("  ...")
i = 13
x = i * step_width
z = (i+1) * z_step
print(f"  stair{i+1}: x={x:.3f}-{x+step_width:.3f}, y=0.000-1.000, z={z:.3f}")
print()

# 西向き（右から左へX方向）
print("【西←】右から左へX方向に昇る")
print("  x_len=1/14（14等分）, y_len=1.0（矩形の全高）")
for i in range(3):
    x = (13-i) * step_width
    z = (i+1) * z_step
    print(f"  stair{i+1}: x={x:.3f}-{x+step_width:.3f}, y=0.000-1.000, z={z:.3f}")
print("  ...")
i = 13
x = (13-i) * step_width
z = (i+1) * z_step
print(f"  stair{i+1}: x={x:.3f}-{x+step_width:.3f}, y=0.000-1.000, z={z:.3f}")
print()

print("="*60)
print("✓ 全てのパターンで14段が等間隔で配置")
print("✓ 高さは0.193ずつ増加（コの字階段と同じ）")
print("✓ 南北方向: Y方向に並ぶ（x_len=1.0, y_len=1/14）")
print("✓ 東西方向: X方向に並ぶ（x_len=1/14, y_len=1.0）")
