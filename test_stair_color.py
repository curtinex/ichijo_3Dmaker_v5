import sys
sys.path.insert(0, r'c:\Users\curtin\Documents\DX\3_DMM生成AI\自由課題\deploy\ichijo_3Dmaker_v5_deploy\ichijo_3Dmaker_v5\ichijo_core_check')

from pathlib import Path
import json

# ui_helpersモジュールを直接インポート（他のモジュールをインポートせずに）
import importlib.util
spec = importlib.util.spec_from_file_location("ui_helpers", r"c:\Users\curtin\Documents\DX\3_DMM生成AI\自由課題\deploy\ichijo_3Dmaker_v5_deploy\ichijo_3Dmaker_v5\ichijo_core_check\ichijo_core\ui_helpers.py")
ui_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui_helpers)

# テストデータ
test_data = {
    'walls': [
        {'id': 'w1', 'start': [0, 0], 'end': [5, 0], 'height': 2.4, 'thickness': 0.12}
    ],
    'stairs': [
        {'name': 'test_stair', 'position': [2.5, 1.0, 0.0], 'size': [1.0, 0.3, 0.2], 'rotation': 0, 'color': 'Tan'}
    ]
}

# JSON作成
json_path = Path('test_stair.json')
json_path.write_text(json.dumps(test_data))

# HTML生成
html_path = Path('test_stair.html')
ui_helpers.generate_3d_viewer_html(json_path, html_path)

print('HTML generated successfully')

# 生成されたHTMLから階段色の部分を抽出して確認
html_content = html_path.read_text(encoding='utf-8')
if 'const stairColor = 0xd2b48c' in html_content:
    print('✅ 階段の色が正しく設定されています: 0xd2b48c (Tan/床と同じ)')
elif '0x8B4513' in html_content:
    print('❌ 古い色が使われています: 0x8B4513 (Walnut brown)')
else:
    print('⚠️ 階段の色定義が見つかりませんでした')
    # stairColorを含む行を抽出
    for line in html_content.split('\n'):
        if 'stairColor' in line:
            print(f'Found line: {line.strip()}')
