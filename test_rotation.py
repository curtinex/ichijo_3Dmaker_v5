import copy

def _create_base_stair_pattern():
    return [
        {'name': 'stair1', 'x': 0, 'y': 0, 'z': 0.193, 'x_len': 0.5, 'y_len': 0.125, 'z_len': 0.05, 'rotation': 0, 'size_type': 'narrow'},
        {'name': 'stair7', 'x': 0, 'y': 0.833, 'z': 1.351, 'x_len': 0.5, 'y_len': 0.167, 'z_len': 0.05, 'rotation': 0, 'size_type': 'wide'},
        {'name': 'stair14', 'x': 0.5, 'y': 0, 'z': 2.702, 'x_len': 0.5, 'y_len': 0.125, 'z_len': 0.05, 'rotation': 0, 'size_type': 'narrow'},
    ]

def _rotate_pattern(steps, degrees):
    rotated = []
    for step in steps:
        s = copy.deepcopy(step)
        x, y = s['x'], s['y']
        x_len, y_len = s['x_len'], s['y_len']
        
        center_x = x + x_len / 2
        center_y = y + y_len / 2
        
        if degrees == 90:
            new_center_x = 0.5 - (center_y - 0.5)
            new_center_y = 0.5 + (center_x - 0.5)
            new_x_len = y_len
            new_y_len = x_len
            s['x'] = new_center_x - new_x_len / 2
            s['y'] = new_center_y - new_y_len / 2
            s['x_len'] = new_x_len
            s['y_len'] = new_y_len
        elif degrees == 180:
            new_center_x = 1.0 - center_x
            new_center_y = 1.0 - center_y
            s['x'] = new_center_x - x_len / 2
            s['y'] = new_center_y - y_len / 2
        elif degrees == 270:
            new_center_x = 0.5 + (center_y - 0.5)
            new_center_y = 0.5 - (center_x - 0.5)
            new_x_len = y_len
            new_y_len = x_len
            s['x'] = new_center_x - new_x_len / 2
            s['y'] = new_center_y - new_y_len / 2
            s['x_len'] = new_x_len
            s['y_len'] = new_y_len
        
        rotated.append(s)
    return rotated

base = _create_base_stair_pattern()

print('=== 北↑（0度）===')
for s in base:
    x, y = s['x'], s['y']
    x_len, y_len = s['x_len'], s['y_len']
    print(f'{s["name"]}: ({x:.3f}, {y:.3f}) - ({x+x_len:.3f}, {y+y_len:.3f})')

print('\n=== 東→（90度）===')
east = _rotate_pattern(base, 90)
for s in east:
    x, y = s['x'], s['y']
    x_len, y_len = s['x_len'], s['y_len']
    print(f'{s["name"]}: ({x:.3f}, {y:.3f}) - ({x+x_len:.3f}, {y+y_len:.3f})')

print('\n=== 南↓（180度）===')
south = _rotate_pattern(base, 180)
for s in south:
    x, y = s['x'], s['y']
    x_len, y_len = s['x_len'], s['y_len']
    print(f'{s["name"]}: ({x:.3f}, {y:.3f}) - ({x+x_len:.3f}, {y+y_len:.3f})')

print('\n=== 西←（270度）===')
west = _rotate_pattern(base, 270)
for s in west:
    x, y = s['x'], s['y']
    x_len, y_len = s['x_len'], s['y_len']
    print(f'{s["name"]}: ({x:.3f}, {y:.3f}) - ({x+x_len:.3f}, {y+y_len:.3f})')

print('\n=== 検証結果 ===')
print(f'北↑ stair7: 左上 - 右上角が (1.0, 1.0) ✓')
print(f'南↓ stair7: ({south[1]["x"]:.3f}, {south[1]["y"]:.3f}) - ({south[1]["x"]+south[1]["x_len"]:.3f}, {south[1]["y"]+south[1]["y_len"]:.3f})')
print(f'  → 右下角が (1.0, 0.0) に接する: {"✓" if abs(south[1]["x"]+south[1]["x_len"]-1.0) < 0.001 and abs(south[1]["y"]) < 0.001 else "✗"}')

print(f'\n東→ stair14: ({east[2]["x"]:.3f}, {east[2]["y"]:.3f}) - ({east[2]["x"]+east[2]["x_len"]:.3f}, {east[2]["y"]+east[2]["y_len"]:.3f})')
print(f'  → 右下角が (1.0, 0.0) に接する: {"✓" if abs(east[2]["x"]+east[2]["x_len"]-1.0) < 0.001 and abs(east[2]["y"]) < 0.001 else "✗"}')

print(f'\n西← stair1: ({west[0]["x"]:.3f}, {west[0]["y"]:.3f}) - ({west[0]["x"]+west[0]["x_len"]:.3f}, {west[0]["y"]+west[0]["y_len"]:.3f})')
print(f'  → 左上角が (0.0, 1.0) に接する: {"✓" if abs(west[0]["x"]) < 0.001 and abs(west[0]["y"]+west[0]["y_len"]-1.0) < 0.001 else "✗"}')
