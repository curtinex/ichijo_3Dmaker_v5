#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlitアプリ: 図面(PDF/JPG/PNG) → 壁線抽出 → 3D(JSON) → Blender用スクリプト生成

使い方:
  1) 下のコマンドで起動
     streamlit run streamlit_app.py
  2) PDF/画像をアップロード → パラメータを調整 → [変換を実行]
  3) 生成されたJSON/Blenderスクリプト/可視化をダウンロード
"""

# ichijo_coreのインストールチェックと自動インストール
import subprocess
import sys
import os

def install_ichijo_core():
    """Streamlit Cloud用: ichijo_coreをGitHubからインストール"""
    print("→ Checking ichijo_core installation...")
    
    # 既にインストール済みで正常にインポートできるかチェック
    try:
        import ichijo_core
        print(f"✓ ichijo_core already installed and importable")
        print(f"  Location: {ichijo_core.__file__}")
        print(f"  Version: {ichijo_core.__version__}")
        return True, None
    except Exception as e:
        print(f"→ ichijo_core not available: {type(e).__name__}: {e}")
        print("→ Proceeding with installation...")
    
    # Streamlit Cloudのsecretsからトークンを取得
    try:
        import streamlit as st_temp
        print(f"→ Checking for GITHUB_TOKEN in secrets...")
        
        if not hasattr(st_temp, 'secrets'):
            error_msg = "Streamlit secrets not available"
            print(f"✗ {error_msg}")
            return False, error_msg
        
        if 'GITHUB_TOKEN' not in st_temp.secrets:
            error_msg = "GITHUB_TOKEN not found in Streamlit secrets"
            print(f"✗ {error_msg}")
            return False, error_msg
        
        token = st_temp.secrets['GITHUB_TOKEN']
        token_preview = token[:8] + "..." if len(token) > 8 else "***"
        print(f"✓ GITHUB_TOKEN found: {token_preview}")
        
        # 一時ディレクトリを作成（インストール先）
        import tempfile
        target_dir = tempfile.mkdtemp(prefix="ichijo_core_")
        print(f"→ Created target directory: {target_dir}")
        
        # sys.pathに追加（パッケージをインポート可能にする）
        if target_dir not in sys.path:
            sys.path.insert(0, target_dir)
            print(f"✓ Added to sys.path: {target_dir}")
        
        # コミットハッシュを使用（最新版 - null bytes修正版）
        commit_hash = "d18749b"
        install_url = f"git+https://{token}@github.com/curtinex/ichijo_core.git@{commit_hash}"
        print(f"→ Installing from: git+https://***@github.com/curtinex/ichijo_core.git@{commit_hash}")
        
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", target_dir, "--force-reinstall", "--no-cache-dir", install_url],
            capture_output=True,
            text=True,
            timeout=300  # 5分タイムアウト
        )
        
        if result.returncode == 0:
            print("✓ ichijo_core installed successfully")
            print(f"STDOUT: {result.stdout[-500:]}")  # 最後の500文字
            
            # インポートキャッシュを無効化（新しいパスを認識させる）
            import importlib
            importlib.invalidate_caches()
            print("✓ Import caches invalidated")
            
            # インストール後、実際にインポートできるか確認
            try:
                import ichijo_core
                print(f"✓ ichijo_core successfully imported from: {ichijo_core.__file__}")
                return True, None
            except Exception as import_error:
                error_msg = f"Installation succeeded but import failed: {type(import_error).__name__}: {str(import_error)}"
                print(f"✗ {error_msg}")
                import traceback
                traceback.print_exc()
                return False, error_msg
        else:
            error_msg = f"pip install failed (exit code {result.returncode})"
            print(f"✗ {error_msg}")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return False, f"{error_msg}\n\nSTDERR:\n{result.stderr[:1000]}"
    except subprocess.TimeoutExpired:
        error_msg = "Installation timed out after 5 minutes"
        print(f"✗ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg

# アプリ起動時に一度だけインストール
success, error_detail = install_ichijo_core()
if not success:
    import streamlit as st
    st.error("❌ ichijo_core のインストールに失敗しました")
    if error_detail:
        with st.expander("🔍 エラー詳細を表示"):
            st.code(error_detail)
    st.info("""
    **トラブルシューティング:**
    
    1. Streamlit Cloud の **Settings → Secrets** で `GITHUB_TOKEN` が正しく設定されているか確認
    2. トークンの形式: `GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"`
    3. トークンの権限: Contents (Read-only), ichijo_core リポジトリへのアクセス権
    4. アプリを再起動してみてください
    """)
    st.stop()

import io
import re
import time
import json
import math
from pathlib import Path
from datetime import datetime
import zipfile

import numpy as np
import streamlit as st
import fitz  # PyMuPDF (for page count)
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

# 共通: 表示用画像を作成して表示スケールを返すヘルパー
def _prepare_display_from_pil(pil_img, max_width=800):
    orig_w, orig_h = pil_img.size
    target_w = max_width if orig_w > max_width else orig_w
    scale = target_w / orig_w
    target_h = int(orig_h * scale)
    display_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return display_img, scale, orig_w, orig_h

def _prepare_display_from_bytes(image_bytes, max_width=800):
    img = Image.open(io.BytesIO(image_bytes))
    return _prepare_display_from_pil(img, max_width=max_width)


# 共通: 表示座標(px) -> 元画像(px)
def _display_to_original(display_x, display_y, display_scale):
    return int(display_x / display_scale), int(display_y / display_scale)


# 共通: 表示座標(px) -> メートル座標(m)
def _display_to_meter(display_x, display_y, display_scale, orig_img_height, margin, scale_px, min_x, min_y):
    orig_x, orig_y = _display_to_original(display_x, display_y, display_scale)
    meter_x = (orig_x - margin) / scale_px + min_x
    meter_y = (orig_img_height - orig_y - margin) / scale_px + min_y
    return meter_x, meter_y, orig_x, orig_y

# ichijo_core から全モジュールをインポート（必須）
try:
    from ichijo_core.pdf_to_image import pdf_to_image
    from ichijo_core.refine_from_image import refine_floor_plan_from_image
    from ichijo_core.extract_walls_to_3d_v2 import process_image_to_3d
    from ichijo_core.visualize_3d_walls import visualize_3d_walls
    from ichijo_core.auto_merge_walls import WallAutoMerger
    from ichijo_core.geometry_utils import (
        calc_distance as _calc_distance,
        calc_angle_diff as _calc_angle_diff,
        wall_angle_deg as _wall_angle_deg,
        angle_diff_deg as _angle_diff_deg,
        determine_line_direction as _determine_line_direction,
    )
    from ichijo_core.furniture_utils import (
        FURNITURE_HEIGHT_OPTIONS,
        FURNITURE_COLOR_OPTIONS,
    )
except ImportError as e:
    st.error(f"❌ ichijo_core パッケージが見つかりません: {e}")
    st.info("このアプリケーションを実行するには ichijo_core パッケージが必要です。")
    st.stop()

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# 窓カタログ（外部ファイルがあれば読み込む。なければ組み込みのデフォルトを使用）
WINDOW_CATALOG_PATH = BASE_DIR / "window_catalog.json"
try:
    if WINDOW_CATALOG_PATH.exists():
        with open(WINDOW_CATALOG_PATH, 'r', encoding='utf-8') as f:
            WINDOW_CATALOG = json.load(f)
    else:
        # デフォルトカタログ（簡易）
        WINDOW_CATALOG = {
            "JF2020/JK2020": {"height": 603, "base": 1541},
            "JF2030/JK2030": {"height": 906, "base": 1238},
            "JF2042/JK2042": {"height": 1175, "base": 969}
        }
except Exception:
    WINDOW_CATALOG = {
        "JF2020/JK2020": {"height": 603, "base": 1541},
        "JF2030/JK2030": {"height": 906, "base": 1238},
        "JF2042/JK2042": {"height": 1175, "base": 969}
    }


def _save_uploaded_file(uploaded_file, dst_path: Path) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst_path


def _generate_3d_viewer_html(json_path: Path, out_path: Path, with_lights: bool = False) -> Path:
    """Three.js HTMLビューアを自動生成（JSON内容を直接埋め込み）
    
    Args:
        json_path: JSONファイルのパス
        out_path: 出力HTMLのパス
        with_lights: True の場合、天井とスポットライトを表示
    """
    # JSONファイルを読み込んで内容を埋め込む
    with open(json_path, 'r', encoding='utf-8') as f:
        json_content = f.read()
    
    html_template = '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>間取り図 3Dビューア</title>
    <style>
        body { margin: 0; overflow: hidden; font-family: sans-serif; }
        #container { width: 100vw; height: 100vh; }
        #info {
            position: absolute; top: 10px; left: 10px;
            background: rgba(0,0,0,0.7); color: white;
            padding: 10px; border-radius: 5px; font-size: 14px;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="info">
        <strong>間取り図 3Dビューア</strong><br>
        初期化中...
    </div>
    <div id="container"></div>

    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        const container = document.getElementById('container');
        const info = document.getElementById('info');

        try {
            info.innerHTML = '<strong>間取り図 3Dビューア</strong><br>読込中...';

            // シーン・カメラ・レンダラー初期化
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(BACKGROUND_COLOR_PLACEHOLDER);

            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(15, 15, 15);

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            // ソフトシャドウを使用して境界を自然にする
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);

            // OrbitControls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            // ライト
            AMBIENT_LIGHT_PLACEHOLDER
            DIRECTIONAL_LIGHT_PLACEHOLDER

            // グリッド
            const gridHelper = new THREE.GridHelper(50, 50, 0x888888, 0xcccccc);
            scene.add(gridHelper);

            // JSON データ（埋め込み）
            const wallsData = JSON_DATA_PLACEHOLDER;
            
            let walls = wallsData.walls || [];
            
            // 3D表示用に壁データを2倍にスケーリング
            walls = walls.map(wall => ({
                ...wall,
                start: [wall.start[0] * 2, wall.start[1] * 2],
                end: [wall.end[0] * 2, wall.end[1] * 2],
                height: wall.height * 2,
                thickness: wall.thickness * 2,
                base_height: (wall.base_height || 0) * 2,
                windows: wall.windows ? wall.windows.map(w => ({
                    ...w,
                    position: w.position * 2,
                    width: w.width * 2,
                    height: w.height * 2
                })) : []
            }));
            
            // 照明データも2倍にスケーリング
            if (wallsData.metadata && wallsData.metadata.lights) {
                wallsData.metadata.lights = wallsData.metadata.lights.map(light => ({
                    ...light,
                    position: [
                        light.position[0] * 2,
                        light.position[1] * 2,
                        light.position[2] * 2
                    ]
                }));
            }
            
            // 図面の中心を計算してオフセット適用
            let offsetX = 0, offsetY = 0;
            // min/max をこのスコープで利用できるように宣言
            let minX = 0, maxX = 0, minY = 0, maxY = 0;
            if (walls.length > 0) {
                const allX = walls.flatMap(w => [w.start[0], w.end[0]]);
                const allY = walls.flatMap(w => [w.start[1], w.end[1]]);
                minX = Math.min(...allX);
                maxX = Math.max(...allX);
                minY = Math.min(...allY);
                maxY = Math.max(...allY);
                offsetX = (minX + maxX) / 2;
                offsetY = (minY + maxY) / 2;
            }
            
            info.innerHTML = `<strong>間取り図 3Dビューア</strong><br>壁数: ${walls.length}<br>マウス: 回転・拡大縮小・移動`;

            // 壁マテリアル
            const wallMaterial = new THREE.MeshStandardMaterial({
                color: 0xffffff,
                emissive: 0xffffff,
                emissiveIntensity: EMISSIVE_INTENSITY_PLACEHOLDER,
                roughness: 0.7,
                metalness: 0.1
            });

            walls.forEach(wall => {
                    const x1 = wall.start[0];
                    const y1 = wall.start[1];
                    const x2 = wall.end[0];
                    const y2 = wall.end[1];
                    const length = Math.sqrt((x2-x1)**2 + (y2-y1)**2);
                    const centerX = (x1 + x2) / 2;
                    const centerY = (y1 + y2) / 2;
                    
                    // base_heightを考慮したZ座標計算（窓対応）
                    const baseHeight = wall.base_height || 0;
                    const centerZ = baseHeight + (wall.height / 2);

                    // BoxGeometry (length, height, thickness)
                    const geometry = new THREE.BoxGeometry(length, wall.height, wall.thickness);
                    const mesh = new THREE.Mesh(geometry, wallMaterial);

                    // 位置（中心オフセットを適用して原点付近に配置、Y座標を反転して鏡像を修正）
                    mesh.position.set(centerX - offsetX, centerZ, -(centerY - offsetY));

                    // 回転（XZ平面上の角度、Y座標反転に合わせて調整）
                    const angle = Math.atan2(-(y2 - y1), x2 - x1);
                    mesh.rotation.y = angle;

                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    scene.add(mesh);
                });

                // 床生成
                // floorsが定義されていればカスタム床、なければ自動生成
                if (wallsData.floors && wallsData.floors.length > 0) {
                    // カスタム床を描画（2倍スケーリング適用）
                    wallsData.floors.forEach((floorData, idx) => {
                        const x1 = floorData.x1 * 2;
                        const y1 = floorData.y1 * 2;
                        const x2 = floorData.x2 * 2;
                        const y2 = floorData.y2 * 2;
                        
                        const floorW = Math.abs(x2 - x1);
                        const floorD = Math.abs(y2 - y1);
                        const centerX = (x1 + x2) / 2;
                        const centerY = (y1 + y2) / 2;
                        
                        const floorGeometry = new THREE.BoxGeometry(floorW, 0.1, floorD);
                        const floorMaterial = new THREE.MeshStandardMaterial({ color: 0xd2b48c });
                        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
                        floor.position.set(centerX - offsetX, -0.05, -(centerY - offsetY));
                        floor.receiveShadow = true;
                        scene.add(floor);
                    });
                } else if (walls.length > 0) {
                    // 自動床生成（壁の範囲から計算）
                    const allX = walls.flatMap(w => [w.start[0], w.end[0]]);
                    const allY = walls.flatMap(w => [w.start[1], w.end[1]]);
                    const minX = Math.min(...allX) - 1;
                    const maxX = Math.max(...allX) + 1;
                    const minY = Math.min(...allY) - 1;
                    const maxY = Math.max(...allY) + 1;

                    const floorW = maxX - minX;
                    const floorD = maxY - minY;
                    const floorGeometry = new THREE.BoxGeometry(floorW, 0.1, floorD);
                    const floorMaterial = new THREE.MeshStandardMaterial({ color: 0xd2b48c });
                    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
                    // 床もオフセットを適用してY座標を反転
                    floor.position.set((minX + maxX) / 2 - offsetX, -0.05, -((minY + maxY) / 2 - offsetY));
                    floor.receiveShadow = true;
                    scene.add(floor);
                }
                    
                WITH_LIGHTS_PLACEHOLDER

                // 家具オブジェクトを表示
                const furniture = wallsData.furniture || [];
                furniture.forEach((item, idx) => {
                    const width = item.dimensions.width * 2;
                    const depth = item.dimensions.depth * 2;
                    const height = item.dimensions.height * 2;
                    
                    const x = item.position[0] * 2;
                    const y = item.position[1] * 2;
                    
                    // 家具の色（JSONから取得、デフォルトは茶色）
                    const furnitureColor = item.color_three_js ? parseInt(item.color_three_js, 16) : 0x8B4513;
                    
                    // 家具のジオメトリとマテリアル
                    const furnitureGeometry = new THREE.BoxGeometry(width, height, depth);
                    const furnitureMaterial = new THREE.MeshStandardMaterial({
                        color: furnitureColor,
                        roughness: 0.8,
                        metalness: 0.2
                    });
                    const furnitureMesh = new THREE.Mesh(furnitureGeometry, furnitureMaterial);
                    
                    // 位置設定（中心オフセット適用、Y座標反転）
                    furnitureMesh.position.set(
                        x - offsetX,
                        height / 2,  // 床から高さの半分の位置
                        -(y - offsetY)
                    );
                    
                    furnitureMesh.castShadow = true;
                    furnitureMesh.receiveShadow = true;
                    scene.add(furnitureMesh);
                });


            // アニメーションループ
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();

            // リサイズ対応
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });

        } catch (err) {
            info.innerHTML = `<strong>エラー発生</strong><br>${err.message}<br><small>コンソールを確認してください</small>`;
            console.error('3Dビューアエラー:', err);
        }
    </script>
</body>
</html>'''
    
    # 照明機能の有無によってコードを切り替え
    if with_lights:
        # 照明付きバージョン：白壁、黒背景、スポットライト
        emissive_intensity = '0.05'
        background_color = '0x000000'
        ambient_light_code = 'const ambientLight = new THREE.AmbientLight(0xffffff, 0.1);\n\n            scene.add(ambientLight);'
        directional_light_code = '// ディレクショナルライトなし（スポットライトのみ）'
        
        lights_code = '''
                    // スポットライト配置
                    const lights = wallsData.metadata?.lights || [];
                    
                    // 天井生成
                    const avgWallHeight = walls.reduce((sum, w) => sum + w.height, 0) / walls.length || 2.7;
                    // floorW/floorD はフロア生成時に定義されない場合があるため、ここでは図面の範囲から算出する
                    const _floorW = (typeof floorW !== 'undefined') ? floorW : (maxX - minX);
                    const _floorD = (typeof floorD !== 'undefined') ? floorD : (maxY - minY);
                    const ceilingGeometry = new THREE.BoxGeometry(_floorW, 0.1, _floorD);
                    const ceilingMaterial = new THREE.MeshStandardMaterial({ color: 0xf5f5f0 });
                    const ceiling = new THREE.Mesh(ceilingGeometry, ceilingMaterial);
                    ceiling.position.set((minX + maxX) / 2 - offsetX, avgWallHeight - 0.05, -((minY + maxY) / 2 - offsetY));
                    ceiling.receiveShadow = true;
                    scene.add(ceiling);
                    
                    // スポットライト配置（入力座標を (x, y, z) として解釈）
                    // ここでは 2D 座標系の (x, y) -> 3D の x,z 平面へマップし、z 成分を高さに使う
                    lights.forEach((light, idx) => {
                        // `照明を配置` UI は position = [meter_x, light_z, meter_y] を保存します
                        // ここではその順 (X, height, Y) を使ってマッピングします
                        // 要求: 照明の水平座標は (0,0) に固定し、高さのみ正の値を使用する
                        const lx = 0;
                        const lz = 0;
                        // position[1] は UI 上の高さ (m)。存在しない場合は天井高さを使用。
                        const ly = Math.abs(light.position[1] || avgWallHeight);

                        // スポットライト本体
                        const spotLight = new THREE.SpotLight(0xffffff, 1.5);
                        spotLight.position.set(lx, ly, lz);
                        spotLight.target.position.set(lx, 0, lz);
                        spotLight.angle = Math.PI / 5;
                        spotLight.penumbra = 0.4;
                        spotLight.decay = 2;
                        spotLight.distance = 20;
                        spotLight.castShadow = true;
                        // シャドウ品質向上
                        spotLight.shadow.mapSize.width = 2048;
                        spotLight.shadow.mapSize.height = 2048;
                        spotLight.shadow.bias = -0.0005;
                        scene.add(spotLight);
                        scene.add(spotLight.target);

                        // 照明位置を示す円柱オブジェクト（天井に設置）
                        const cylinderGeometry = new THREE.CylinderGeometry(0.1, 0.1, 0.3, 16);
                        const cylinderMaterial = new THREE.MeshStandardMaterial({ 
                            color: 0xffff00,
                            emissive: 0xffff00,
                            emissiveIntensity: 0.5
                        });
                        const cylinder = new THREE.Mesh(cylinderGeometry, cylinderMaterial);
                        // 天井の下面に設置（天井高さ - 0.15m）
                        // 円柱（視覚表示）は天井近くの中心に配置
                        cylinder.position.set(0, avgWallHeight - 0.15, 0);
                        scene.add(cylinder);
                    });
'''
    else:
        # 通常バージョン：白壁、ライトグレー背景、ディレクショナルライトあり
        emissive_intensity = '0.4'
        background_color = '0xf0f0f0'
        ambient_light_code = 'const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);\n\n            scene.add(ambientLight);'
        # デフォルトのディレクショナルライトを原点の真上に固定して影が斜めにならないようにする
        directional_light_code = '''const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
            // 真上から照らす（シーン原点を照らす）
            dirLight.position.set(0, 50, 0);
            // ターゲットを原点に設定
            dirLight.target = new THREE.Object3D();
            dirLight.target.position.set(0, 0, 0);
            scene.add(dirLight.target);
            dirLight.castShadow = true;
            // シャドウ品質向上: 解像度とバイアスを設定
            dirLight.shadow.mapSize.width = 2048;
            dirLight.shadow.mapSize.height = 2048;
            dirLight.shadow.bias = -0.0005;
            // オルソカメラの範囲はフロアサイズに合わせて自動調整（floorW/floorD が存在すれば利用）
            if (dirLight.shadow && dirLight.shadow.camera && dirLight.shadow.camera.isOrthographicCamera) {
                const s = Math.max((typeof floorW !== 'undefined' ? floorW : 10), (typeof floorD !== 'undefined' ? floorD : 10));
                dirLight.shadow.camera.left = -s;
                dirLight.shadow.camera.right = s;
                dirLight.shadow.camera.top = s;
                dirLight.shadow.camera.bottom = -s;
                dirLight.shadow.camera.near = 0.5;
                dirLight.shadow.camera.far = 200;
            }
            scene.add(dirLight);'''
        lights_code = '                    // 照明機能なし\n\n'
    
    # プレースホルダーを置換
    html_content = html_template.replace('JSON_DATA_PLACEHOLDER', json_content)
    html_content = html_content.replace('EMISSIVE_INTENSITY_PLACEHOLDER', emissive_intensity)
    html_content = html_content.replace('BACKGROUND_COLOR_PLACEHOLDER', background_color)
    html_content = html_content.replace('AMBIENT_LIGHT_PLACEHOLDER', ambient_light_code)
    html_content = html_content.replace('DIRECTIONAL_LIGHT_PLACEHOLDER', directional_light_code)
    html_content = html_content.replace('WITH_LIGHTS_PLACEHOLDER', lights_code)
    out_path.write_text(html_content, encoding='utf-8')
    
    return out_path


# 以下の関数は ichijo_core.geometry_utils からインポート済み:
# - _calc_distance, _calc_angle_diff, _wall_angle_deg, _angle_diff_deg, _determine_line_direction

# 以下の定数は ichijo_core.furniture_utils からインポート済み:
# - FURNITURE_HEIGHT_OPTIONS, FURNITURE_COLOR_OPTIONS


def _snap_to_grid(rect_pixel, json_data, scale, grid_size=0.45):
    """
    ピクセル座標の四角形をメートル座標に変換（グリッドスナップなし）
    
    Args:
        rect_pixel: (x1, y1, x2, y2) ピクセル座標の四角形
        json_data: JSON壁データ（座標変換用）
        scale: 可視化スケール (px/m)
        grid_size: グリッド間隔（m）※未使用
    
    Returns:
        (x_start_m, y_start_m, width_m, depth_m): メートル座標での配置情報
    """
    x1_px, y1_px, x2_px, y2_px = rect_pixel
    
    # 座標変換パラメータを取得
    all_x = [w['start'][0] for w in json_data['walls']] + [w['end'][0] for w in json_data['walls']]
    all_y = [w['start'][1] for w in json_data['walls']] + [w['end'][1] for w in json_data['walls']]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    margin = 50
    img_height = int((max_y - min_y) * scale) + 2 * margin
    
    # ピクセル → メートル座標に変換（スナップなし、四角形範囲をそのまま使用）
    x1_m = (min(x1_px, x2_px) - margin) / scale + min_x
    y1_m = (img_height - max(y1_px, y2_px) - margin) / scale + min_y
    x2_m = (max(x1_px, x2_px) - margin) / scale + min_x
    y2_m = (img_height - min(y1_px, y2_px) - margin) / scale + min_y
    
    # 幅と奥行きを計算（選択範囲そのまま）
    width_m = x2_m - x1_m
    depth_m = y2_m - y1_m
    
    return x1_m, y1_m, width_m, depth_m


def _add_furniture_to_json(json_data, height_m, color_name, x_start, y_start, width, depth):
    """
    JSONに家具オブジェクトを追加
    
    Args:
        json_data: 既存のJSON
        height_m: 家具の高さ（メートル）
        color_name: 色の名前（FURNITURE_COLOR_OPTIONSのキー）
        x_start, y_start: 配置開始座標（メートル）
        width, depth: 幅と奥行き（メートル）
    """
    import copy
    
    updated_data = copy.deepcopy(json_data)
    
    # furnitureキーがなければ作成
    if 'furniture' not in updated_data:
        updated_data['furniture'] = []
    
    color_info = FURNITURE_COLOR_OPTIONS[color_name]
    
    # 家具オブジェクトを追加
    furniture_obj = {
        "type": f"家具（{color_name}）",
        "position": [x_start + width/2, y_start + depth/2],  # 中心座標
        "dimensions": {
            "width": width,
            "depth": depth,
            "height": height_m
        },
        "bounds": {
            "x_start": x_start,
            "y_start": y_start,
            "x_end": x_start + width,
            "y_end": y_start + depth
        },
        "color": color_name,
        "color_rgb": color_info["rgb"],
        "color_three_js": color_info["three_js"],
        "description": f"{color_name}の家具（高さ{height_m*100:.0f}cm）"
    }
    
    updated_data['furniture'].append(furniture_obj)
    
    # メタデータ更新
    if 'metadata' not in updated_data:
        updated_data['metadata'] = {}
    updated_data['metadata']['furniture_count'] = len(updated_data['furniture'])
    
    return updated_data


def _select_best_wall_pair_from_4(walls):
    """4本の壁から結合すべき最適な2本を選択
    
    端点が4つ矩形に含まれた場合、縦横を判定して最も近い平行な壁のペアを選ぶ。
    
    Args:
        walls: 壁のリスト（4本を想定）
    
    Returns:
        [wall1, wall2]: 選択された2本の壁、または None
    """
    import math
    
    if len(walls) < 2:
        return None
    
    # 各壁を縦方向/横方向に分類
    vertical_walls = []
    horizontal_walls = []
    
    for wall in walls:
        x1, y1 = wall['start']
        x2, y2 = wall['end']
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # x差が小さい → 縦方向の壁、y差が小さい → 横方向の壁
        if dx < dy:
            vertical_walls.append(wall)
        else:
            horizontal_walls.append(wall)
    
    # デバッグ情報を追加
    try:
        v_ids = [w.get('id', '?') for w in vertical_walls]
        h_ids = [w.get('id', '?') for w in horizontal_walls]
        append_debug(f"Wall classification: vertical={v_ids} ({len(vertical_walls)}), horizontal={h_ids} ({len(horizontal_walls)})")
    except:
        pass
    
    # 縦方向の壁が2本ある場合：各壁の平均X座標の差を計算
    dX = float('inf')
    if len(vertical_walls) >= 2:
        wall1 = vertical_walls[0]
        wall2 = vertical_walls[1]
        
        avg_x1 = (wall1['start'][0] + wall1['end'][0]) / 2
        avg_x2 = (wall2['start'][0] + wall2['end'][0]) / 2
        
        dX = abs(avg_x1 - avg_x2)
    
    # 横方向の壁が2本ある場合：各壁の平均Y座標の差を計算
    dY = float('inf')
    if len(horizontal_walls) >= 2:
        wall1 = horizontal_walls[0]
        wall2 = horizontal_walls[1]
        
        avg_y1 = (wall1['start'][1] + wall1['end'][1]) / 2
        avg_y2 = (wall2['start'][1] + wall2['end'][1]) / 2
        
        dY = abs(avg_y1 - avg_y2)
    
    # デバッグ情報を追加
    try:
        append_debug(f"Distance comparison: dX={dX:.3f}, dY={dY:.3f}")
    except:
        pass
    
    # dXとdYの小さい方が結合すべき壁の組み合わせ
    # 両方ともinfの場合（縦2本も横2本もない）はNoneを返す
    if dX == float('inf') and dY == float('inf'):
        try:
            append_debug(f"ERROR: No valid pair found (both dX and dY are inf)")
        except:
            pass
        return None
    
    if dX < dY:
        # 縦方向の壁2本を選択
        result = vertical_walls[:2] if len(vertical_walls) >= 2 else None
        try:
            result_ids = [w.get('id', '?') for w in result] if result else []
            append_debug(f"Selected vertical walls: {result_ids}")
        except:
            pass
        return result
    else:
        # 横方向の壁2本を選択
        result = horizontal_walls[:2] if len(horizontal_walls) >= 2 else None
        try:
            result_ids = [w.get('id', '?') for w in result] if result else []
            append_debug(f"Selected horizontal walls: {result_ids}")
        except:
            pass
        return result


def _add_line_to_json(json_data, p1, p2, wall_height=None, scale=50):
    """四角形選択から線を追加（2点から自動判定した方向で線を生成）"""
    import copy
    
    # 元データを保護するためディープコピー
    updated_data = copy.deepcopy(json_data)
    walls = updated_data['walls']
    
    # 既存の壁から平均厚さを取得
    thicknesses = [w.get('thickness', 0.12) for w in walls if 'thickness' in w]
    default_thickness = sum(thicknesses) / len(thicknesses) if thicknesses else 0.12
    
    # 既存の壁から平均高さを取得（指定がない場合）
    if wall_height is None:
        heights = [w.get('height', 2.4) for w in walls if 'height' in w]
        wall_height = sum(heights) / len(heights) if heights else 2.4
    
    # 四角形の座標を計算
    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
    
    # 可視化画像のメートル座標変換パラメータを取得
    all_x = [w['start'][0] for w in json_data['walls']] + [w['end'][0] for w in json_data['walls']]
    all_y = [w['start'][1] for w in json_data['walls']] + [w['end'][1] for w in json_data['walls']]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    margin = 50
    img_height = int((max_y - min_y) * scale) + 2 * margin
    
    # ピクセル座標 → メートル座標に変換
    x1_m = (x1 - margin) / scale + min_x
    y1_m = (img_height - y1 - margin) / scale + min_y
    x2_m = (x2 - margin) / scale + min_x
    y2_m = (img_height - y2 - margin) / scale + min_y
    
    # 方向を判定
    direction = _determine_line_direction(p1, p2)
    
    # 新しい壁線を生成
    if direction == "vertical":
        # 縦線：x座標を四角形の中央に固定、y座標は上下端
        x_center = (x1_m + x2_m) / 2
        start_pt = [x_center, min(y1_m, y2_m)]
        end_pt = [x_center, max(y1_m, y2_m)]
    else:  # horizontal
        # 横線：y座標を四角形の中央に固定、x座標は左右端
        y_center = (y1_m + y2_m) / 2
        start_pt = [min(x1_m, x2_m), y_center]
        end_pt = [max(x1_m, x2_m), y_center]
    
    # 線の長さを計算
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    length = round(math.sqrt(dx**2 + dy**2), 3)
    
    # 新しい壁のIDを生成（IDが文字列の場合も対応）
    try:
        max_id = max([int(w['id']) for w in walls], default=0)
    except (ValueError, TypeError):
        max_id = 0
    new_id = max_id + 1
    
    # 新しい壁オブジェクトを作成（既存の壁と同じ構造）
    new_wall = {
        'id': new_id,
        'start': [round(start_pt[0], 3), round(start_pt[1], 3)],
        'end': [round(end_pt[0], 3), round(end_pt[1], 3)],
        'height': round(wall_height, 3),  # 既存の壁と同じ高さ
        'base_height': 0.0,  # 通常の壁は床から
        'length': length,
        'thickness': round(default_thickness, 3),  # 既存の壁と同じ厚さ
        'source': 'added'  # 手動追加の壁として記録
    }
    
    # 壁を追加
    walls.append(new_wall)
    
    # メタデータ更新
    updated_data['metadata']['total_walls'] = len(walls)
    
    return updated_data, direction, new_wall


def _point_in_rect(point, rect):
    """点が四角形内にあるかチェック（可視化画像のピクセル座標）"""
    x, y = point
    x_min, y_min = rect['left'], rect['top']
    x_max, y_max = rect['left'] + rect['width'], rect['top'] + rect['height']
    return x_min <= x <= x_max and y_min <= y <= y_max


def _line_intersects_rect(x1, y1, x2, y2, rect, tolerance=20):
    """線分が四角形と交差または近接しているかチェック（拡張版）"""
    x_min = rect['left'] - tolerance
    y_min = rect['top'] - tolerance
    x_max = rect['left'] + rect['width'] + tolerance
    y_max = rect['top'] + rect['height'] + tolerance
    
    # 1. 端点が四角形内にある
    if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or \
       (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
        return True
    
    # 2. 線分が四角形の辺と交差するか（簡易判定）
    # 線分が四角形を完全に横断している場合
    if (x1 < x_min and x2 > x_max) or (x2 < x_min and x1 > x_max) or \
       (y1 < y_min and y2 > y_max) or (y2 < y_min and y1 > y_max):
        return True
    
    # 3. 四角形が線分の間にある
    if (min(x1, x2) <= x_max and max(x1, x2) >= x_min) and \
       (min(y1, y2) <= y_max and max(y1, y2) >= y_min):
        return True
    
    return False


def _wall_in_rect(wall, rect, scale, margin, img_height, min_x, min_y, max_x, max_y):
    """壁線が四角形選択範囲内または近接しているかチェック（拡張版）"""
    # 壁線のメートル座標をピクセル座標に変換（visualize_3d_wallsと同じロジック）
    x1_px = int((wall['start'][0] - min_x) * scale) + margin
    y1_px = img_height - (int((wall['start'][1] - min_y) * scale) + margin)
    x2_px = int((wall['end'][0] - min_x) * scale) + margin
    y2_px = img_height - (int((wall['end'][1] - min_y) * scale) + margin)
    
    # 線分が四角形と交差または近接しているかチェック（許容範囲20ピクセル）
    return _line_intersects_rect(x1_px, y1_px, x2_px, y2_px, rect, tolerance=20)


def _filter_walls_strictly_in_rect(walls, rect, scale, margin, img_height, min_x, min_y, max_x, max_y):
    """
    四角形範囲内に完全に含まれる壁線のみを返す（精密フィルタリング）
    交差や近接ではなく、両端点が四角形内にある線のみを抽出
    """
    filtered_walls = []
    
    x_rect_min = rect['left']
    x_rect_max = rect['left'] + rect['width']
    y_rect_min = rect['top']
    y_rect_max = rect['top'] + rect['height']
    
    for wall in walls:
        # ピクセル座標に変換
        x1_px = int((wall['start'][0] - min_x) * scale) + margin
        y1_px = img_height - (int((wall['start'][1] - min_y) * scale) + margin)
        x2_px = int((wall['end'][0] - min_x) * scale) + margin
        y2_px = img_height - (int((wall['end'][1] - min_y) * scale) + margin)
        
        # 両端点が四角形内にあるかチェック（許容値なし）
        if (x_rect_min <= x1_px <= x_rect_max and y_rect_min <= y1_px <= y_rect_max and
            x_rect_min <= x2_px <= x_rect_max and y_rect_min <= y2_px <= y_rect_max):
            filtered_walls.append(wall)
    
    return filtered_walls


def _line_intersects_rect(x1, y1, x2, y2, rect_or_left, rect_top=None, rect_right=None, rect_bottom=None, tolerance=0):
    """線分が四角形と交差または近接しているかチェック。
    互換性を持たせるため、以下の呼び出し形式の両方を受け入れます:
      - _line_intersects_rect(x1,y1,x2,y2, rect_dict, tolerance=...)
      - _line_intersects_rect(x1,y1,x2,y2, left, top, right, bottom)
    """

    # rect_or_left が辞書で渡された場合は展開
    if rect_top is None and isinstance(rect_or_left, dict):
        rect = rect_or_left
        rect_left = rect.get('left', 0)
        rect_top = rect.get('top', 0)
        rect_right = rect_left + rect.get('width', 0)
        rect_bottom = rect_top + rect.get('height', 0)
    else:
        # 数値で渡された場合
        rect_left = rect_or_left

    # 拡張判定: tolerance を四角形に適用して拡大する
    x_min = rect_left - tolerance
    y_min = rect_top - tolerance
    x_max = rect_right + tolerance
    y_max = rect_bottom + tolerance

    # 1. 端点が四角形内にある
    if (x_min <= x1 <= x_max and y_min <= y1 <= y_max) or \
       (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
        return True

    # ヘルパ: 線分交差判定（2D、閉区間）
    def _on_segment(ax, ay, bx, by, cx, cy):
        return min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by)

    def _orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _segments_intersect(a1, a2, b1, b2):
        ax1, ay1 = a1
        ax2, ay2 = a2
        bx1, by1 = b1
        bx2, by2 = b2
        o1 = _orient(ax1, ay1, ax2, ay2, bx1, by1)
        o2 = _orient(ax1, ay1, ax2, ay2, bx2, by2)
        o3 = _orient(bx1, by1, bx2, by2, ax1, ay1)
        o4 = _orient(bx1, by1, bx2, by2, ax2, ay2)

        if o1 == 0 and _on_segment(ax1, ay1, ax2, ay2, bx1, by1):
            return True
        if o2 == 0 and _on_segment(ax1, ay1, ax2, ay2, bx2, by2):
            return True
        if o3 == 0 and _on_segment(bx1, by1, bx2, by2, ax1, ay1):
            return True
        if o4 == 0 and _on_segment(bx1, by1, bx2, by2, ax2, ay2):
            return True

        return (o1 * o2 < 0) and (o3 * o4 < 0)

    # 2. 線分と四角形の4辺が交差しているか厳密に判定
    rect_edges = [
        ((x_min, y_min), (x_max, y_min)),  # top
        ((x_max, y_min), (x_max, y_max)),  # right
        ((x_max, y_max), (x_min, y_max)),  # bottom
        ((x_min, y_max), (x_min, y_min)),  # left
    ]

    seg_a = ((x1, y1), (x2, y2))
    for edge in rect_edges:
        if _segments_intersect(seg_a[0], seg_a[1], edge[0], edge[1]):
            return True

    # 3. 上記で判定できなければ重なりなし
    return False


def _filter_walls_by_endpoints_in_rect(walls, rect, scale, margin, img_height, min_x, min_y, max_x, max_y, tolerance=0, debug=False):
    """
    四角形範囲内に端点があるか、線分が四角形と交差する壁線を返す（窓追加モード用）
    
    Args:
        tolerance: 端点判定の許容範囲（ピクセル）デフォルト0px（誤検出防止）
        debug: Trueの場合、詳細なデバッグ情報を返す
    """
    filtered_walls = []
    debug_info = []
    
    x_rect_min = rect['left']
    x_rect_max = rect['left'] + rect['width']
    y_rect_min = rect['top']
    y_rect_max = rect['top'] + rect['height']
    
    if debug:
        debug_info.append(f"四角形範囲: X[{x_rect_min:.1f}, {x_rect_max:.1f}], Y[{y_rect_min:.1f}, {y_rect_max:.1f}]")
        debug_info.append(f"四角形サイズ: {rect['width']:.1f} x {rect['height']:.1f} px")
        if tolerance > 0:
            debug_info.append(f"許容範囲: ±{tolerance}px")
        debug_info.append(f"検証対象の壁数: {len(walls)}本")
        debug_info.append("---")
    
    for wall in walls:
        # ピクセル座標に変換
        x1_px = int((wall['start'][0] - min_x) * scale) + margin
        y1_px = img_height - (int((wall['start'][1] - min_y) * scale) + margin)
        x2_px = int((wall['end'][0] - min_x) * scale) + margin
        y2_px = img_height - (int((wall['end'][1] - min_y) * scale) + margin)
        
        # 端点が四角形内にあるかチェック
        start_in_rect = (
            x_rect_min - tolerance <= x1_px <= x_rect_max + tolerance and 
            y_rect_min - tolerance <= y1_px <= y_rect_max + tolerance
        )
        end_in_rect = (
            x_rect_min - tolerance <= x2_px <= x_rect_max + tolerance and 
            y_rect_min - tolerance <= y2_px <= y_rect_max + tolerance
        )
        
        # 線分が四角形と交差するかチェック
        intersects = _line_intersects_rect(
            x1_px, y1_px, x2_px, y2_px,
            x_rect_min, y_rect_min, x_rect_max, y_rect_max
        )
        
        matched = start_in_rect or end_in_rect or intersects
        
        if debug:
            wall_id = wall.get('id', '?')
            reasons = []
            if start_in_rect:
                reasons.append("始点✓")
            if end_in_rect:
                reasons.append("終点✓")
            if intersects and not (start_in_rect or end_in_rect):
                reasons.append("交差✓")
            
            reason_str = ",".join(reasons) if reasons else "範囲外✗"
            
            # 検出された壁のみ、または全壁を表示（最初の50本まで）
            if matched or len(debug_info) < 60:  # ヘッダー行+最大50壁
                debug_info.append(
                    f"壁#{wall_id}: start({x1_px:.0f},{y1_px:.0f}) end({x2_px:.0f},{y2_px:.0f}) → {reason_str}"
                )
        
        if matched:
            filtered_walls.append(wall)
    
    if debug:
        return filtered_walls, debug_info
    return filtered_walls


def _find_collinear_chains(walls_in_selection, distance_threshold=0.3, angle_threshold=15):
    """一直線上に並んだ連結壁線のチェーンを検出（3本以上の線を結合可能に）"""
    if len(walls_in_selection) < 2:
        return []
    
    # 各壁線間の接続情報を構築
    connections = {}  # {wall_id: [(connected_wall_id, connection_type, distance), ...]}
    
    for i, wall1 in enumerate(walls_in_selection):
        wall1_id = wall1['id']
        if wall1_id not in connections:
            connections[wall1_id] = []
        
        for j, wall2 in enumerate(walls_in_selection):
            if i >= j:
                continue
            
            wall2_id = wall2['id']
            if wall2_id not in connections:
                connections[wall2_id] = []
            
            # 角度が同一直線上かチェック
            angle_diff = _calc_angle_diff(wall1, wall2)
            if angle_diff >= angle_threshold:
                continue
            
            # 4つの端点組み合わせをチェック
            endpoint_pairs = [
                (wall1['end'], wall2['start'], 'end-start'),
                (wall1['end'], wall2['end'], 'end-end'),
                (wall1['start'], wall2['start'], 'start-start'),
                (wall1['start'], wall2['end'], 'start-end'),
            ]
            
            for p1, p2, connection_type in endpoint_pairs:
                distance = _calc_distance(p1, p2)
                if distance < distance_threshold:
                    connections[wall1_id].append((wall2_id, connection_type, distance))
                    # 逆方向の接続も記録
                    reverse_type = connection_type.split('-')[::-1]
                    reverse_type = f"{reverse_type[0]}-{reverse_type[1]}"
                    connections[wall2_id].append((wall1_id, reverse_type, distance))
                    break
    
    # 連結チェーンを検出（DFS）
    visited = set()
    chains = []
    
    def build_chain(start_wall_id, current_chain, visited_in_chain):
        """再帰的にチェーンを構築"""
        if start_wall_id in visited_in_chain:
            return
        
        visited_in_chain.add(start_wall_id)
        current_chain.append(start_wall_id)
        
        # 接続された壁線を探索
        if start_wall_id in connections:
            for connected_id, conn_type, dist in connections[start_wall_id]:
                if connected_id not in visited_in_chain:
                    build_chain(connected_id, current_chain, visited_in_chain)
    
    # 各壁線からチェーンを構築
    for wall in walls_in_selection:
        wall_id = wall['id']
        if wall_id not in visited:
            chain = []
            visited_in_chain = set()
            build_chain(wall_id, chain, visited_in_chain)
            
            if len(chain) >= 2:  # 2本以上のチェーン
                visited.update(chain)
                # チェーン内の壁線オブジェクトを取得
                chain_walls = [w for w in walls_in_selection if w['id'] in chain]
                chains.append(chain_walls)
    
    return chains


def _find_mergeable_walls(walls_in_selection, distance_threshold=0.3, angle_threshold=15):
    """選択範囲内で結合可能な壁線ペアまたはチェーンを探す（3本以上も対応）"""
    candidates = []
    
    # まず、一直線上の連結チェーンを検出（3本以上対応）
    chains = _find_collinear_chains(walls_in_selection, distance_threshold, angle_threshold)
    
    # チェーンを結合候補として追加
    for chain in chains:
        if len(chain) >= 2:
            # チェーンの最初と最後の壁線を取得
            first_wall = chain[0]
            last_wall = chain[-1]
            
            # チェーン全体の端点を決定
            # 最初の壁線と最後の壁線の端点から、最も離れた2点を選ぶ
            all_endpoints = [
                first_wall['start'],
                first_wall['end'],
                last_wall['start'],
                last_wall['end']
            ]
            
            max_dist = 0
            chain_start = None
            chain_end = None
            
            for i, p1 in enumerate(all_endpoints):
                for j, p2 in enumerate(all_endpoints):
                    if i >= j:
                        continue
                    dist = _calc_distance(p1, p2)
                    if dist > max_dist:
                        max_dist = dist
                        chain_start = p1
                        chain_end = p2
            
            # チェーン全体の平均角度差を計算
            total_angle_diff = 0
            for i in range(len(chain) - 1):
                total_angle_diff += _calc_angle_diff(chain[i], chain[i+1])
            avg_angle_diff = total_angle_diff / (len(chain) - 1) if len(chain) > 1 else 0
            
            candidates.append({
                'walls': chain,  # チェーン全体
                'is_chain': True,
                'chain_length': len(chain),
                'distance': max_dist,
                'angle_diff': avg_angle_diff,
                'new_start': chain_start,
                'new_end': chain_end,
                'confidence': 1.0
            })
    
    # 個別のペアも検出（従来の機能を維持）
    for i, wall1 in enumerate(walls_in_selection):
        for j, wall2 in enumerate(walls_in_selection):
            if i >= j:
                continue
            
            # 4つの端点組み合わせをチェック
            connections = [
                (wall1['end'], wall2['start'], 'end-start', wall1['end'], wall2['end']),
                (wall1['end'], wall2['end'], 'end-end', wall1['end'], wall2['start']),
                (wall1['start'], wall2['start'], 'start-start', wall1['end'], wall2['end']),
                (wall1['start'], wall2['end'], 'start-end', wall1['end'], wall2['start']),
            ]
            
            for p1, p2, connection_type, new_p1, new_p2 in connections:
                distance = _calc_distance(p1, p2)
                angle_diff = _calc_angle_diff(wall1, wall2)
                
                # 距離と角度の両方でチェック（180度回転も同一方向として扱う）
                if distance < distance_threshold and angle_diff < angle_threshold:
                    candidates.append({
                        'wall1': wall1,
                        'wall2': wall2,
                        'is_chain': False,
                        'distance': distance,
                        'angle_diff': angle_diff,
                        'connection': connection_type,
                        'new_start': new_p1,
                        'new_end': new_p2,
                        'confidence': 1.0 - (distance / distance_threshold)
                    })
    
    # 信頼度でソート
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    return candidates


def _merge_walls_in_json(json_data, merge_pairs):
    """JSONデータ内の壁線を結合（ディープコピーで元データを保護、3本以上のチェーンにも対応）"""
    import copy
    # 元のデータを変更しないようにディープコピーを作成
    updated_data = copy.deepcopy(json_data)
    walls = updated_data['walls']
    
    for pair in merge_pairs:
        # チェーン結合（3本以上）の場合
        if pair.get('is_chain', False) and 'walls' in pair:
            chain_walls = pair['walls']
            if len(chain_walls) < 2:
                continue
            
            # 最初の壁線を残して更新、他は削除
            first_wall_id = chain_walls[0]['id']
            other_wall_ids = [w['id'] for w in chain_walls[1:]]
            
            # 最初の壁線を探して座標を更新
            for wall in walls:
                if wall['id'] == first_wall_id:
                    wall['start'] = pair['new_start']
                    wall['end'] = pair['new_end']
                    
                    # 長さを再計算
                    dx = wall['end'][0] - wall['start'][0]
                    dy = wall['end'][1] - wall['start'][1]
                    wall['length'] = round(math.sqrt(dx**2 + dy**2), 3)
                    break
            
            # チェーン内の他の壁線を削除
            walls[:] = [w for w in walls if w['id'] not in other_wall_ids]
        
        # 通常のペア結合（2本）の場合
        elif 'wall1' in pair and 'wall2' in pair:
            wall1_id = pair['wall1']['id']
            wall2_id = pair['wall2']['id']

            # 接続タイプに応じて新しい端点を決定（pair に含まれる wall オブジェクトを参照）
            conn = pair.get('connection')
            w1 = pair['wall1']
            w2 = pair['wall2']

            if conn == 'end-start':
                new_start = w1.get('start')
                new_end = w2.get('end')
            elif conn == 'end-end':
                new_start = w1.get('start')
                new_end = w2.get('start')
            elif conn == 'start-start':
                new_start = w1.get('end')
                new_end = w2.get('end')
            elif conn == 'start-end':
                new_start = w1.get('end')
                new_end = w2.get('start')
            else:
                # 不明な接続タイプの場合は pair に保持されている new_start/new_end を優先
                new_start = pair.get('new_start')
                new_end = pair.get('new_end')

            # wall1 を探して更新
            for wall in walls:
                if wall['id'] == wall1_id:
                    if new_start is not None:
                        wall['start'] = new_start
                    if new_end is not None:
                        wall['end'] = new_end

                    # 長さを再計算
                    dx = wall['end'][0] - wall['start'][0]
                    dy = wall['end'][1] - wall['start'][1]
                    wall['length'] = round(math.sqrt(dx**2 + dy**2), 3)
                    break

            # wall2 を削除
            walls[:] = [w for w in walls if w['id'] != wall2_id]
    
    # メタデータ更新
    updated_data['metadata']['total_walls'] = len(walls)
    return updated_data


def _delete_walls_in_json(json_data, wall_ids_to_delete):
    """JSONデータ内の指定された壁線を削除"""
    import copy
    # 元のデータを変更しないようにディープコピーを作成
    updated_data = copy.deepcopy(json_data)
    walls = updated_data['walls']
    
    # 削除対象の壁IDセット化（高速な検索のため）
    delete_ids = set(wall_ids_to_delete)
    
    # 指定された壁を削除
    walls[:] = [w for w in walls if w['id'] not in delete_ids]
    
    # メタデータ更新
    updated_data['metadata']['total_walls'] = len(walls)
    return updated_data


def _add_window_walls(json_data, wall1, wall2, window_height, base_height, room_height, window_model=None, window_height_mm=None):
    """
    窓で分断された2本の壁の間に、床側と天井側の壁を追加
    
    Args:
        json_data: 元のJSONデータ
        wall1, wall2: 窓で分断された2本の壁
        window_height: 窓の高さ（m）
        base_height: 床から窓下端までの高さ（m）
        room_height: 部屋の天井高さ（m）
    
    Returns:
        更新されたJSONデータ、追加された壁のリスト
    """
    import copy
    updated_data = copy.deepcopy(json_data)
    walls = updated_data['walls']
    
    # 2本の壁の端点から窓の位置を計算
    # wall1の端点とwall2の端点で最も近い組み合わせを見つける
    endpoints = [
        (wall1['start'], wall2['start']),
        (wall1['start'], wall2['end']),
        (wall1['end'], wall2['start']),
        (wall1['end'], wall2['end']),
    ]
    
    min_dist = float('inf')
    window_start = None
    window_end = None
    
    for p1, p2 in endpoints:
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if dist < min_dist:
            min_dist = dist
            window_start = p1
            window_end = p2
    
    # 既存の壁から平均厚さを取得
    thicknesses = [w.get('thickness', 0.12) for w in walls if 'thickness' in w]
    default_thickness = sum(thicknesses) / len(thicknesses) if thicknesses else 0.12
    
    # 新しい壁のIDを生成
    try:
        max_id = max([int(w['id']) for w in walls], default=0)
    except (ValueError, TypeError):
        max_id = 0
    
    added_walls = []
    
    # 床側の壁を追加（床〜窓下端）
    floor_wall = {
        'id': max_id + 1,
        'start': [round(window_start[0], 3), round(window_start[1], 3)],
        'end': [round(window_end[0], 3), round(window_end[1], 3)],
        'height': round(base_height, 3),
        'base_height': 0.0,  # 床から
        'length': round(min_dist, 3),
        'thickness': round(default_thickness, 3),
        'source': 'window_added',
        'window_model': window_model,
        'window_height_mm': window_height_mm,
        'window_base_m': round(base_height, 3),
        'window_base_mm': int(round(base_height * 1000))
    }
    walls.append(floor_wall)
    added_walls.append(floor_wall)
    
    # 天井側の壁を追加（窓上端〜天井）
    ceiling_height = room_height - (base_height + window_height)
    ceiling_wall = {
        'id': max_id + 2,
        'start': [round(window_start[0], 3), round(window_start[1], 3)],
        'end': [round(window_end[0], 3), round(window_end[1], 3)],
        'height': round(ceiling_height, 3),
        'base_height': round(base_height + window_height, 3),  # 窓上端から
        'length': round(min_dist, 3),
        'thickness': round(default_thickness, 3),
        'source': 'window_added',
        'window_model': window_model,
        'window_height_mm': window_height_mm,
        'window_base_m': round(base_height + window_height, 3),
        'window_base_mm': int(round((base_height + window_height) * 1000))
    }
    walls.append(ceiling_wall)
    added_walls.append(ceiling_wall)
    
    # メタデータ更新
    updated_data['metadata']['total_walls'] = len(walls)
    
    return updated_data, added_walls


def _find_closest_wall_to_point(walls, point_px, scale, margin, img_height, min_x, min_y, max_x, max_y):
    """ポイントから最も近い壁を見つける"""
    min_distance = float('inf')
    closest_wall = None
    
    # ポイントをメートル座標に変換
    point_m = [
        (point_px[0] - margin) / scale + min_x,
        (img_height - point_px[1] - margin) / scale + min_y
    ]
    
    for wall in walls:
        # 壁の両端点を取得（安全にアンパック）
        try:
            start = wall.get('start')
            end = wall.get('end')
            
            # 配列形式であることを確認
            if not isinstance(start, (list, tuple)) or not isinstance(end, (list, tuple)):
                continue
            if len(start) < 2 or len(end) < 2:
                continue
            
            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])
        except (TypeError, ValueError, KeyError):
            continue
        
        # ポイントから線分までの最短距離を計算
        # 線分上の最近点を見つける
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # 点と点の距離
            distance = math.sqrt((point_m[0] - x1)**2 + (point_m[1] - y1)**2)
        else:
            # 線分上での最近点を計算
            t = max(0, min(1, ((point_m[0] - x1) * dx + (point_m[1] - y1) * dy) / (dx**2 + dy**2)))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            distance = math.sqrt((point_m[0] - closest_x)**2 + (point_m[1] - closest_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_wall = wall
    
    return closest_wall, min_distance


def main():
    st.set_page_config(page_title="一条工務店 CAD図面3D化アプリ", layout="wide")
    st.title("一条工務店 CAD図面3D化アプリ")
    st.caption("アップロードした図面は一時的な処理にのみ使用し、データベースに保存されることはありません。")
    
    # 固定画像幅（自動結合と手動編集で統一）
    DISPLAY_IMAGE_WIDTH = 800

    # セッション状態の初期化（結果の永続化）
    if "processed" not in st.session_state:
        st.session_state.processed = False
    for key in [
        "out_dir", "refined_img", "refined_name", "refined_bytes", "json_bytes", "json_name",
        "viz_bytes", "viz_name", "viewer_html_bytes", "viewer_html_name",
        "zip_bytes", "zip_name", "merged_json_bytes", "merged_json_name", "merged_viz_bytes", "merged_viz_name",
        "merged_processed"
    ]:
        st.session_state.setdefault(key, None)
    if "merged_processed" not in st.session_state:
        st.session_state.merged_processed = False
    if "workflow_step" not in st.session_state:
        st.session_state.workflow_step = 1  # 1:読み込み, 2:スケール校正, 3:手動編集
        # debug: record initialization
        st.session_state.setdefault('debug_log', []).append(f"init: set workflow_step=1")
    if "merge_choice" not in st.session_state:
        st.session_state.merge_choice = "自動壁結合を実行"
    if "viz_scale" not in st.session_state:
        st.session_state.viz_scale = 100  # 可視化スケール（メートル→ピクセル）固定値

    # 2D可視化スケールを固定値に設定
    viz_scale = 100

    # ============= ステップバー表示 =============
    st.markdown("### 📍 作業フロー")
    col_steps = st.columns(3, gap="small")
    step_names = ["① 画像読み込み", "② スケール校正", "③ 手動編集"]
    step_status = []

    # 永続デバッグログ初期化とヘルパー
    if 'debug_log' not in st.session_state:
        st.session_state['debug_log'] = []
    # (internal debug_log retained in session_state; no UI display)

    # --- debug helper: callback to set workflow step reliably ---
    def _set_workflow_step(n: int):
        st.session_state.setdefault('debug_log', []).append(f"callback: set_workflow {n} (was {st.session_state.get('workflow_step')})")
        st.session_state.workflow_step = n
        st.session_state.setdefault('debug_log', []).append(f"callback: workflow_step now {st.session_state.get('workflow_step')}")

    # manual debug button removed

    def append_debug(msg: str):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.setdefault('debug_log', []).append(f"{ts} {msg}")
        except Exception:
            # append に失敗しても処理継続
            pass
    
    for i in range(3):
        if i + 1 < st.session_state.workflow_step:
            step_status.append("✅")
        elif i + 1 == st.session_state.workflow_step:
            step_status.append("▶️")
        else:
            step_status.append("")
    
    for col, status, name in zip(col_steps, step_status, step_names):
        with col:
            st.write(f"{status} {name}")
    
    st.divider()

    # ============= ステップ1: 画像読み込み =============
    with st.expander("Step 1：画像読み込み", expanded=(st.session_state.workflow_step == 1)):
        st.markdown("## ステップ ① 画像読み込み")

        # 最初のステップの説明
        st.info(
            "**ステップ①** 間取り図のPDF（またはJPG/PNG）をアップロードしてください。\n\n"
            # "• **PDF**: 図面アプリから出力したPDFファイル\n\n"
            # "• **JPG/PNG**: スキャン画像やスクリーンショット"
        )

        uploaded = st.file_uploader("図面PDF/画像をアップロード", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=False)
    
        # PDFの場合、ページ数を確認してページ選択UIを表示
        page_number = 0  # デフォルト
        if uploaded is not None and uploaded.name.lower().endswith('.pdf'):
            try:
                # 一時的にPDFを保存してページ数を取得
                temp_pdf = io.BytesIO(uploaded.getvalue())
                doc = fitz.open(stream=temp_pdf, filetype="pdf")
                total_pages = len(doc)
                doc.close()
                
                if total_pages > 1:
                    st.info(f"📄 このPDFは {total_pages} ページあります。処理するページを選択してください。")
                    page_number = st.selectbox(
                        "処理するページを選択",
                        options=list(range(total_pages)),
                        format_func=lambda x: f"ページ {x + 1}",
                        help="PDFの何ページ目を処理するか選択します（0始まり）"
                    )
                else:
                    st.info("📄 このPDFは 1 ページです。")
            except Exception as e:
                st.warning(f"PDFページ数の取得に失敗しました: {e}")
        
        # if uploaded is None and not st.session_state.processed:
        #     st.info("PDF/JPG/PNGをアップロードしてください。")
        
        # アップロード後のパラメータ設定を折りたたみで表示
        if uploaded is not None or st.session_state.processed:
            with st.expander("⚙️ **パラメータ設定（必要に応じて修正）**", expanded=False):
                # 3列レイアウト
                col1, col2, col3 = st.columns(3)
                
                # 列1: 除外範囲設定
                with col1:
                    st.subheader("✂️ 除外範囲")
                    st.markdown("#### 外枠・ロゴ除外")
                    with st.expander("💡 使い方", expanded=False):
                        st.markdown(
                            "**調整方法:**\n\n"
                            "- PDFの外枠、タイトル、ロゴを除外\n\n"
                            "- 中心部の間取り図だけを処理\n\n"
                            "- 除外が少ない → 値を上げる\n\n"
                            "- 図面が欠ける → 値を下げる"
                        )
                    
                    margin_vertical = st.slider(
                        "上下除外(%)",
                        min_value=0,
                        max_value=30,
                        value=10,
                        step=1,
                        help="上下の外枠・タイトル・ロゴを除外します。"
                    )
                    
                    margin_horizontal = st.slider(
                        "左右除外(%)",
                        min_value=0,
                        max_value=30,
                        value=10,
                        step=1,
                        help="左右の外枠・ロゴを除外します。"
                    )
                
                # 列2: 壁検出パラメータ(黒閾値、最小線幅)
                with col2:
                    st.subheader("🎨 壁パラメータ")
                    
                    #st.markdown("#### 黒線認識の閾値")
                    #with st.expander("💡 調整のコツ", expanded=False):
                    #    st.markdown(
                    #        "**調整方法:**\n\n"
                    #        "1. 最初は190から試す\n\n"
                    #        "2. ノイズが多い → 値を下げる\n\n"
                    #        "3. 薄い線が消える → 値を上げる"
                    #    )
                    
                    #black_threshold = st.slider(
                    #    "黒閾値",
                    #    min_value=100,
                    #    max_value=240,
                    #    value=190,
                    #    step=5,
                    #    help="値を上げる = 薄い線も拾う。値を下げる = 濃い線のみ。"
                    #)
                    black_threshold = 190  # 固定値に変更

                    st.markdown("#### 最小認識 黒線幅")
                    with st.expander("💡 調整のコツ", expanded=False):
                        st.markdown(
                            "**調整方法:**\n\n"
                            "1. 壁が多すぎる → 値を上げる\n\n"
                            "2. 壁が足りない → 値を下げる"
                        )
                    
                    min_thickness = st.slider(
                        "最小線幅(px)",
                        min_value=3,
                        max_value=20,
                        value=10,
                        step=1,
                        help="小さい = 細い線も拾う。大きい = 太い線のみ。DPI高い時は大きく。"
                    )
                
                # 列3: Blender出力スケール(壁高さ)
                with col3:
                    st.subheader("🏗️ 出力スケール")
                    wall_height = st.number_input(
                        "壁(天井)高さ",
                        min_value=0.1,
                        max_value=10.0,
                        value=2.4,
                        step=0.1,
                        help="部屋の天井高さ（床から天井までの高さ）をメートル単位で指定します。一般的な住宅は2.4m程度です。"
                    )
                
                # PDFレンダリングDPI（固定値に変更）
                dpi = 300
                
                # ピクセル→メートル係数: スケール校正済みの値があればそれを使用、なければデフォルト値
                if "json_bytes" in st.session_state and st.session_state.json_bytes:
                    try:
                        json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                        pixel_to_meter = json_data.get("metadata", {}).get("pixel_to_meter", 0.005) or 0.005
                        if pixel_to_meter != 0.005:
                            st.info(f"📐 スケール校正済みの値を使用: pixel_to_meter = {pixel_to_meter:.6f}")
                    except:
                        pixel_to_meter = 0.005
                else:
                    pixel_to_meter = 0.005
            
            # ステップ1: 変換実行ボタン
            col_run, col_skip = st.columns([2, 1])
            with col_run:
                run = st.button("🚀 変換を実行", type="primary", use_container_width=True, key="step1_run")
            with col_skip:
                if st.button("⏭️ スキップ", use_container_width=True, key="step1_skip"):
                    st.session_state.workflow_step = 2
                    st.rerun()
        else:
            # アップロード前のデフォルト値を設定
            dpi = 300
            black_threshold = 190
            min_thickness = 8
            pixel_to_meter = 0.005
            wall_height = 2.4
            margin_vertical = 10
            margin_horizontal = 5
            run = False

        # 途中経過の表示可否（Falseで最終結果のみ表示）
        show_progress = False

        if run:
            if uploaded is None:
                st.error("ファイルが未選択です。先にPDF/画像をアップロードしてください。")
                st.stop()
            # 出力ディレクトリ（タイムスタンプで分離）
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = OUTPUTS_DIR / f"run_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # 入力ファイル保存
            input_suffix = Path(uploaded.name).suffix.lower()
            input_path = out_dir / f"input{input_suffix}"
            _save_uploaded_file(uploaded, input_path)
            if show_progress:
                st.success(f"入力を保存: {input_path}")

            # PDFなら画像へレンダリング
            if input_suffix == ".pdf":
                if show_progress:
                    st.write(f"PDFを画像に変換中… (ページ {page_number + 1})")
                source_image = out_dir / "source.png"
                pdf_to_image(str(input_path), output_path=str(source_image), page_number=page_number, dpi=dpi)
                image_path = source_image
            else:
                image_path = input_path

            if show_progress:
                st.write("壁線抽出とノイズ除去を実行中…")
            try:
                refined_img, refined_path = refine_floor_plan_from_image(
                    str(image_path), 
                    black_threshold=black_threshold, 
                    min_thickness=min_thickness, 
                    remove_corners=False,
                    margin_vertical=margin_vertical,
                    margin_horizontal=margin_horizontal
                )
                refined_path = Path(refined_path)
                
                # refined画像をout_dirに移動
                refined_dest = out_dir / refined_path.name
                if refined_path.exists() and refined_path != refined_dest:
                    refined_path.rename(refined_dest)
                    refined_path = refined_dest
                    if show_progress:
                        st.success(f"壁線抽出完了: {refined_path.name}")
            except Exception as e:
                st.error(f"壁線抽出でエラー: {e}")
                return

            # 3D JSON 生成
            if show_progress:
                st.write("3D座標JSONを生成中…")
            json_path = out_dir / "walls_3d.json"
            try:
                result = process_image_to_3d(str(refined_path), str(json_path), wall_height=wall_height, pixel_to_meter=pixel_to_meter)
                if result is None:
                    st.error("❌ 3D座標の生成に失敗しました。")
                    st.warning("⚠️ 原因: refined画像から壁線が検出できませんでした。")
                    st.info(
                        "**対策:**\n\n"
                        "- 最小線幅を大きくする（8-12px推奨）\n\n"
                        "- 黒閾値を調整する（190-210推奨）\n\n"
                        "- DPIを上げる（400-600推奨）"
                    )
                    # デバッグ用：refined画像が実際に存在するか確認
                    if refined_path.exists():
                        st.info(f"✓ refined画像は生成されています: {refined_path.name}")
                    return
            except Exception as e:
                st.error(f"❌ 3D座標生成でエラー: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

            # 2D可視化
            if show_progress:
                st.write("2D可視化画像を生成中…")
            viz_path = out_dir / "visualization.png"
            try:
                canvas = visualize_3d_walls(str(json_path), str(viz_path), scale=int(viz_scale), wall_color=(0, 0, 0), bg_color=(255, 255, 255))
            except Exception as e:
                canvas = None
                st.warning(f"可視化の生成に失敗しました: {e}")

            # Three.js HTMLビューア生成
            if show_progress:
                st.write("3DビューアHTMLを生成中…")
            viewer_html = out_dir / "viewer_3d.html"
            try:
                _generate_3d_viewer_html(json_path, viewer_html)
            except Exception as e:
                st.error(f"3DビューアHTML生成でエラー: {e}")
                return

            # セッションに結果を保存（ダウンロードで再実行されても残す）
            st.session_state.out_dir = str(out_dir)
            st.session_state.refined_img = refined_img
            st.session_state.refined_name = refined_path.name
            st.session_state.refined_bytes = refined_path.read_bytes()
            st.session_state.json_bytes = json_path.read_bytes()
            st.session_state.json_name = json_path.name
            st.session_state.viewer_html_bytes = viewer_html.read_bytes()
            st.session_state.viewer_html_name = viewer_html.name
            if canvas is not None and viz_path.exists():
                st.session_state.viz_bytes = viz_path.read_bytes()
                st.session_state.viz_name = viz_path.name
            else:
                st.session_state.viz_bytes = None
                st.session_state.viz_name = None
            
            # ====== 自動結合を実行 ======
            try:
                # デフォルトを 30px に変更して誤結合を抑制
                merge_radius = st.session_state.get("merge_radius", 55)
                merge_angle = st.session_state.get("merge_angle", 15)
                
                merged_json_path = out_dir / "walls_3d_merged.json"
                merged_viz_path = out_dir / "visualization_merged.png"
                
                # (表示削除) 自動結合で使用するパラメータの情報表示を削除しました
                merger = WallAutoMerger(search_radius=merge_radius, angle_tolerance=merge_angle)
                merger.process(str(refined_path), str(json_path), str(merged_json_path))
                
                visualize_3d_walls(
                    str(merged_json_path),
                    str(merged_viz_path),
                    scale=int(viz_scale),
                    wall_color=(0, 0, 0),
                    bg_color=(255, 255, 255)
                )
                
                merged_viewer_path = out_dir / "viewer_3d_merged.html"
                _generate_3d_viewer_html(merged_json_path, merged_viewer_path)
                
                st.session_state.merged_json_bytes = merged_json_path.read_bytes()
                st.session_state.merged_json_name = merged_json_path.name
                if merged_viz_path.exists():
                    st.session_state.merged_viz_bytes = merged_viz_path.read_bytes()
                    st.session_state.merged_viz_name = merged_viz_path.name
                if merged_viewer_path.exists():
                    st.session_state.merged_viewer_html_bytes = merged_viewer_path.read_bytes()
                    st.session_state.merged_viewer_html_name = merged_viewer_path.name
                
                # 結合済みファイルを作業ファイルとして設定
                st.session_state.json_bytes = merged_json_path.read_bytes()
                st.session_state.json_name = merged_json_path.name
                st.session_state.viz_bytes = merged_viz_path.read_bytes()
                st.session_state.viz_name = merged_viz_path.name
                st.session_state.viewer_html_bytes = merged_viewer_path.read_bytes()
                st.session_state.viewer_html_name = merged_viewer_path.name
                st.session_state.merged_processed = True
            except Exception as e:
                st.warning(f"⚠️ 自動結合でエラー: {e}")
                st.session_state.merged_processed = False
            
            # ZIP生成（JSON+抽出PNG+3DビューアHTML）
            try:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(st.session_state.json_name, st.session_state.json_bytes)
                    zf.writestr(st.session_state.viewer_html_name, st.session_state.viewer_html_bytes)
                    if st.session_state.refined_bytes and st.session_state.refined_name:
                        zf.writestr(st.session_state.refined_name, st.session_state.refined_bytes)
                zip_buf.seek(0)
                st.session_state.zip_bytes = zip_buf.getvalue()
                st.session_state.zip_name = f"{out_dir.name}_bundle.zip"
            except Exception:
                st.session_state.zip_bytes = None
                st.session_state.zip_name = None
            st.session_state.processed = True
            st.success("変換が完了しました！")

            # セッションに結果があれば常に表示（ダウンロードでの再実行でも消えない）
            if st.session_state.processed:
                st.session_state.setdefault('debug_log', []).append(
                    f"render: entering processed block (workflow_step={st.session_state.get('workflow_step')}, processed={st.session_state.get('processed')}, viewer_html={'yes' if st.session_state.get('viewer_html_bytes') else 'no'})"
                )
                # 画像を横並びで表示
                col_refined, col_viz = st.columns(2)

                with col_refined:
                    st.subheader("🖼️ 壁線抽出結果（CAD図面参照）")
                    if st.session_state.refined_img is not None:
                        # 画面サイズに応じて自動調整（見切れないようにコンテナ幅に合わせる）
                        st.image(st.session_state.refined_img, clamp=True, use_container_width=True)

                with col_viz:
                    st.subheader("📊 3Dモデル用イメージ")
                    if st.session_state.viz_bytes is not None:
                        # 画面サイズに応じて自動調整（見切れないようにコンテナ幅に合わせる）
                        st.image(st.session_state.viz_bytes, use_container_width=True)

                # 説明文を追加
                st.success(
                    "壁の分断や過不足がある場合はパラメータを再設定して再度変換するか、手動編集で修正可能です。\n\n"
                    "窓は手動編集で追加できます。"
                )

                # 3DビューアHTMLダウンロードボタン
                if st.session_state.viewer_html_bytes:
                    st.download_button(
                        label="3Dモデルをダウンロード",
                        data=st.session_state.viewer_html_bytes,
                        file_name=st.session_state.viewer_html_name,
                        mime="text/html"
                    )

                # ステップ1: 読み取り完了ボタン（左寄せ、押すとStep2へ遷移）
                st.session_state.setdefault('debug_log', []).append("render: before creating step1_complete button")
                # ボタン列幅を広げ、ボタンをコンテナ幅いっぱいに表示して折返しを防止
                col_btn, col_rest = st.columns([3, 7])
                with col_btn:
                    st.button("✅ 読み取り完了", type="primary", key="step1_complete", on_click=_set_workflow_step, args=(2,), use_container_width=True)

    with st.expander("Step 2：スケール校正", expanded=(st.session_state.workflow_step == 2)):
        if st.session_state.workflow_step >= 2 and st.session_state.processed:
            st.divider()
            st.markdown("## ステップ ② スケール校正")
        st.info(
            "壁のサイズを確認してスケールを調整してください(一条CAD図面 1マス = 編集画面 2マス推奨)\n\n"
            "基準となる壁を囲む四角形をつくるように2点クリックして、修正スケール値を入力して実行してください。(スキップ可)"
        )
        
        col_skip_calib = st.container()
        with col_skip_calib:
            if st.button("⏭️ スキップして次へ", use_container_width=True, key="step3_skip"):
                st.session_state.workflow_step = 3
                st.rerun()

        # 2点選択用の簡易編集エリアを即時表示
        # st.markdown("### 🖼️ 画像をクリックして2点を選択")
        st.caption("2点をクリックすると赤丸と四角形を表示し、範囲内の壁を赤色にハイライトします。選択しない場合はスキップで次へ進めます。")

        # 初期化
        if "scale_points" not in st.session_state:
            st.session_state.scale_points = []            # 画像原寸座標で保持
        if "scale_last_click" not in st.session_state:
            st.session_state.scale_last_click = None

        if st.session_state.viz_bytes:
            try:
                import math
                from PIL import ImageDraw

                disp_img, scale_disp, orig_w, orig_h = _prepare_display_from_bytes(
                    st.session_state.viz_bytes, max_width=DISPLAY_IMAGE_WIDTH
                )

                # 壁データ読み込みと座標変換（streamlit_app.pyと同じロジック）
                json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                walls = json_data.get("walls", [])
                scale_px = int(st.session_state.viz_scale)
                margin = 50  # streamlit_app.pyと同じマージン
                all_x, all_y = [], []
                for w in walls:
                    s = w.get("start", [])
                    e = w.get("end", [])
                    if len(s) >= 2 and len(e) >= 2:
                        all_x.extend([s[0], e[0]])
                        all_y.extend([s[1], e[1]])
                if all_x and all_y:
                    min_x, max_x = min(all_x), max(all_x)
                    min_y, max_y = min(all_y), max(all_y)
                    # visualization画像のサイズを計算（Y座標反転を考慮）
                    img_width_calc = int((max_x - min_x) * scale_px) + 2 * margin
                    img_height_calc = int((max_y - min_y) * scale_px) + 2 * margin
                else:
                    min_x = min_y = 0
                    max_x = max_y = 1
                    img_width_calc = orig_w
                    img_height_calc = orig_h

                # オーバーレイ描画用にコピー
                overlay = disp_img.copy()
                draw = ImageDraw.Draw(overlay)

                # 既存ポイントを赤丸で描画（表示座標へスケール）
                for pt in st.session_state.scale_points:
                    dx = int(pt["x"] * scale_disp)
                    dy = int(pt["y"] * scale_disp)
                    draw.ellipse((dx - 6, dy - 6, dx + 6, dy + 6), fill=(255, 0, 0), outline=(255, 0, 0))

                # 2点そろったら四角形と壁のハイライト
                if len(st.session_state.scale_points) == 2:
                    p1, p2 = st.session_state.scale_points
                    rx1, rx2 = sorted([p1["x"], p2["x"]])
                    ry1, ry2 = sorted([p1["y"], p2["y"]])

                    # 表示座標で四角形描画
                    draw.rectangle(
                        (int(rx1 * scale_disp), int(ry1 * scale_disp), int(rx2 * scale_disp), int(ry2 * scale_disp)),
                        outline=(0, 200, 0), width=3
                    )

                    # 四角形内の壁を赤で描画（Y座標反転を考慮）
                    for w in walls:
                        s = w.get("start", [])
                        e = w.get("end", [])
                        if len(s) >= 2 and len(e) >= 2:
                            # メートル→ピクセル変換（Y座標反転を考慮）
                            x1 = int((s[0] - min_x) * scale_px) + margin
                            y1 = img_height_calc - (int((s[1] - min_y) * scale_px) + margin)
                            x2 = int((e[0] - min_x) * scale_px) + margin
                            y2 = img_height_calc - (int((e[1] - min_y) * scale_px) + margin)

                            # 壁の両端が四角形内に含まれるかチェック（誤差5px）
                            tolerance = 5
                            start_in_rect = (rx1 - tolerance <= x1 <= rx2 + tolerance and
                                           ry1 - tolerance <= y1 <= ry2 + tolerance)
                            end_in_rect = (rx1 - tolerance <= x2 <= rx2 + tolerance and
                                         ry1 - tolerance <= y2 <= ry2 + tolerance)
                            
                            if start_in_rect and end_in_rect:
                                # 表示座標で赤線を描画
                                dx1 = int(x1 * scale_disp)
                                dy1 = int(y1 * scale_disp)
                                dx2 = int(x2 * scale_disp)
                                dy2 = int(y2 * scale_disp)
                                draw.line((dx1, dy1, dx2, dy2), fill=(255, 0, 0), width=4)

                    # 四角形内の壁を検出（streamlit_app.pyと同じロジック）
                    walls_in_rect = []
                    for w in walls:
                        s = w.get("start", [])
                        e = w.get("end", [])
                        if len(s) >= 2 and len(e) >= 2:
                            # メートル→ピクセル変換（Y座標反転を考慮）
                            x1 = int((s[0] - min_x) * scale_px) + margin
                            y1 = img_height_calc - (int((s[1] - min_y) * scale_px) + margin)
                            x2 = int((e[0] - min_x) * scale_px) + margin
                            y2 = img_height_calc - (int((e[1] - min_y) * scale_px) + margin)

                            # 壁の両端が四角形内に含まれるかチェック（誤差5px）
                            tolerance = 5
                            start_in_rect = (rx1 - tolerance <= x1 <= rx2 + tolerance and
                                           ry1 - tolerance <= y1 <= ry2 + tolerance)
                            end_in_rect = (rx1 - tolerance <= x2 <= rx2 + tolerance and
                                         ry1 - tolerance <= y2 <= ry2 + tolerance)
                            
                            if start_in_rect and end_in_rect:
                                wall_length_px = math.hypot(x2 - x1, y2 - y1)
                                walls_in_rect.append({
                                    'wall': w,
                                    'id': w.get('id', '?'),
                                    'length_px': wall_length_px,
                                    'start_m': s,
                                    'end_m': e
                                })

                    if walls_in_rect:
                        # 最長の壁を使用
                        target_wall_data = max(walls_in_rect, key=lambda w: w['length_px'])
                        px_distance = target_wall_data['length_px']
                        wall_id = target_wall_data['id']
                        # 2点選択直後（スケール校正/四角形選択）では表示を抑制する
                        if not (len(st.session_state.get('rect_coords', [])) == 2 or len(st.session_state.get('scale_points', [])) == 2):
                            st.success(f"✅ 壁({wall_id})を検出: {px_distance:.1f}px")
                    else:
                        # 同様に2点選択直後はエラーメッセージを表示しない
                        if not (len(st.session_state.get('rect_coords', [])) == 2 or len(st.session_state.get('scale_points', [])) == 2):
                            st.error("❌ 四角形内に完全に含まれる壁が見つかりません")
                        px_distance = None
                        target_wall_data = None

                    # スケール入力と反映
                    if px_distance is not None and target_wall_data is not None:
                        # スケール校正時に2点選択直後は表示を抑制する
                        if not (len(st.session_state.get('rect_coords', [])) == 2 or len(st.session_state.get('scale_points', [])) == 2):
                            # 現在のデータから座標範囲を表示
                            st.markdown("---")
                            st.markdown("### 📊 現在のデータ情報")
                            
                            # 現在のJSONデータから座標範囲を取得
                            json_data_current = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            walls_current = json_data_current.get("walls", [])
                            all_x_current = []
                            all_y_current = []
                            for w in walls_current:
                                if "start" in w and "end" in w:
                                    all_x_current.extend([w["start"][0], w["end"][0]])
                                    all_y_current.extend([w["start"][1], w["end"][1]])
                            
                            if all_x_current and all_y_current:
                                min_x_current = min(all_x_current)
                                max_x_current = max(all_x_current)
                                min_y_current = min(all_y_current)
                                max_y_current = max(all_y_current)
                                width_current = max_x_current - min_x_current
                                height_current = max_y_current - min_y_current
                                
                                st.info(
                                    f"**現在の座標範囲:**\n"
                                    f"- X軸（幅）: {min_x_current:.3f}m ～ {max_x_current:.3f}m → **{width_current:.3f}m** = **{width_current/0.9:.1f}マス**\n"
                                    f"- Y軸（奥行）: {min_y_current:.3f}m ～ {max_y_current:.3f}m → **{height_current:.3f}m** = **{height_current/0.9:.1f}マス**"
                                )
                                
                                # 選択した壁の情報
                                current_wall = target_wall_data['wall']
                                wall_length = current_wall.get('length')
                                wall_height = current_wall.get('height', 2.4)
                                wall_thickness = current_wall.get('thickness', 0.1)
                                st.write(
                                    f"**選択した壁（ID: {target_wall_data['id']}）:**\n\n"
                                    f"- 長さ: {wall_length:.3f}m = {wall_length/0.9:.2f}マス\n\n"
                                    f"- 高さ: {wall_height}m（校正後は2.4m固定）\n\n"
                                    f"- 厚さ: {wall_thickness}m（校正後は0.1m固定）"
                                )
                            
                            st.markdown("---")
                        
                        default_grid = st.session_state.get("step3_grid_input_val", 1.0)
                        grid_count = st.number_input(
                            "この壁は一条工務店CAD図面上で何マス分ですか？ (1マス=0.9m)",
                            min_value=0.1,
                            max_value=100.0,
                            value=float(default_grid),
                            step=0.1,
                            key="step3_grid_input"
                        )
                        st.session_state.step3_grid_input_val = grid_count
                        
                        if st.button("💾 このスケールで更新", type="primary", use_container_width=True, key="step3_apply_scale"):
                            try:
                                # 実測距離（メートル単位）
                                actual_distance_m = grid_count * 0.9  # 1マス = 0.9m = 90cm
                                
                                # 現在の壁の長さ（メートル単位）を取得
                                current_length_m = target_wall_data['wall'].get('length')
                                if current_length_m is None:
                                    dx_m = target_wall_data['end_m'][0] - target_wall_data['start_m'][0]
                                    dy_m = target_wall_data['end_m'][1] - target_wall_data['start_m'][1]
                                    current_length_m = math.sqrt(dx_m**2 + dy_m**2)
                                
                                if current_length_m <= 0:
                                    st.error("❌ 現在の壁長が0mのため再計算できません。別の壁で試してください。")
                                else:
                                    # スケール比率を計算（実測/現在）
                                    scale_ratio = actual_distance_m / current_length_m
                                    
                                    # 現在のJSONを読み込み
                                    import copy
                                    out_dir = Path(st.session_state.out_dir)
                                    json_path = out_dir / st.session_state.json_name
                                    json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                    old_pixel_to_meter = json_data.get("metadata", {}).get("pixel_to_meter", 0.005) or 0.005
                                    
                                    # 新しいpixel_to_meterを計算
                                    new_pixel_to_meter = old_pixel_to_meter * scale_ratio
                                    
                                    st.info(
                                        f"実測 {actual_distance_m:.2f}m / 現在 {current_length_m:.2f}m → 倍率 {scale_ratio:.3f}\n\n"
                                        f"旧pixel_to_meter: {old_pixel_to_meter:.6f} → 新: {new_pixel_to_meter:.6f}"
                                    )
                                    
                                    # 各壁の座標をスケール変換
                                    calibrated_json = copy.deepcopy(json_data)
                                    for wall in calibrated_json.get("walls", []):
                                        if "start" in wall and "end" in wall:
                                            # 座標をスケール変換（X-Y平面のみ）
                                            wall["start"] = [round(wall["start"][0] * scale_ratio, 3),
                                                           round(wall["start"][1] * scale_ratio, 3)]
                                            wall["end"] = [round(wall["end"][0] * scale_ratio, 3),
                                                         round(wall["end"][1] * scale_ratio, 3)]
                                            # 長さを再計算（X-Y平面の長さ）
                                            dx = wall["end"][0] - wall["start"][0]
                                            dy = wall["end"][1] - wall["start"][1]
                                            wall["length"] = round(math.sqrt(dx**2 + dy**2), 3)
                                            
                                            # 高さは2.4m固定（スケール変換しない）
                                            # 一条工務店の標準天井高は2.4mで固定
                                            if "height" not in wall or wall.get("height") != 2.4:
                                                wall["height"] = 2.4
                                            
                                            # 厚さは10cm固定（スケール変換しない）
                                            # 一条工務店の標準壁厚は10cmで固定
                                            wall["thickness"] = 0.1
                                    
                                    # メタデータを更新
                                    calibrated_json.setdefault("metadata", {})["pixel_to_meter"] = new_pixel_to_meter
                                    
                                    # デバッグ情報：校正後の座標範囲を計算
                                    all_x_after = []
                                    all_y_after = []
                                    for w in calibrated_json.get("walls", []):
                                        if "start" in w and "end" in w:
                                            all_x_after.extend([w["start"][0], w["end"][0]])
                                            all_y_after.extend([w["start"][1], w["end"][1]])
                                    
                                    if all_x_after and all_y_after:
                                        min_x_after, max_x_after = min(all_x_after), max(all_x_after)
                                        min_y_after, max_y_after = min(all_y_after), max(all_y_after)
                                        width_after = max_x_after - min_x_after
                                        height_after = max_y_after - min_y_after
                                        st.info(
                                            f"📐 校正後の座標範囲:\n\n"
                                            f"X: {min_x_after:.3f}m ～ {max_x_after:.3f}m（幅 {width_after:.3f}m = {width_after/0.9:.1f}マス）\n\n"
                                            f"Y: {min_y_after:.3f}m ～ {max_y_after:.3f}m（奥行 {height_after:.3f}m = {height_after/0.9:.1f}マス）"
                                        )
                                    
                                    # 保存と再可視化
                                    json_path.write_text(json.dumps(calibrated_json, indent=2, ensure_ascii=False))
                                    st.session_state.json_bytes = json_path.read_bytes()
                                    viz_path = out_dir / st.session_state.viz_name
                                    visualize_3d_walls(
                                        str(json_path),
                                        str(viz_path),
                                        scale=int(st.session_state.viz_scale),
                                        wall_color=(0, 0, 0),
                                        bg_color=(255, 255, 255)
                                    )
                                    if viz_path.exists():
                                        st.session_state.viz_bytes = viz_path.read_bytes()

                                    # 後続用に状態更新
                                    st.session_state.scale_calibration_done = True
                                    st.session_state.scale_points = []
                                    st.session_state.scale_last_click = None
                                    st.session_state.step3_grid_input_val = grid_count
                                    # 手動編集へ遷移
                                    st.session_state.workflow_step = 3
                                    st.success(f"✅ スケールを更新しました。pixel_to_meter = {new_pixel_to_meter:.6f}")
                                    st.info(f"📝 編集済みの壁構成を維持したまま、スケールのみを調整しました")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ スケール更新でエラーが発生しました: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                # クリック受付（表示画像）
                click = streamlit_image_coordinates(overlay, key="step3_calib_click")

                # クリック処理（重複防止）
                if click:
                    cur = (click["x"], click["y"])
                    if st.session_state.scale_last_click != cur:
                        st.session_state.scale_last_click = cur
                        # 表示座標を原寸に戻して保持
                        px_orig, py_orig = _display_to_original(click["x"], click["y"], scale_disp)
                        st.session_state.scale_points.append({"x": px_orig, "y": py_orig})
                        if len(st.session_state.scale_points) > 2:
                            st.session_state.scale_points = st.session_state.scale_points[-2:]
                        st.rerun()

                # 選択状態の表示（非表示に変更）

                col_reset = st.columns(2)[0]
                with col_reset:
                    if st.button("🔄 選択をリセット", use_container_width=True, key="step3_calib_reset"):
                        st.session_state.scale_points = []
                        st.session_state.scale_last_click = None
                        st.rerun()

                # スケール適用済みの案内（遷移は適用時に実施済み）
                if st.session_state.get("scale_calibration_done"):
                    st.success("スケールを適用しました。手動編集に進んでください。")
            except Exception as e:
                st.error(f"スケール校正ビュー表示エラー: {e}")

    # ============= ステップ3: 手動編集 =============
    with st.expander("Step 3：手動編集", expanded=(st.session_state.workflow_step == 3)):
        if st.session_state.workflow_step >= 3 and st.session_state.processed:
            st.divider()
            st.markdown("## ステップ ③ 手動編集")
        # --- 自動結合の現在値表示と再実行ボタン ---
        # 自動結合の手動操作UIは不要のため削除しました。
        # 固定パラメータを使用します: merge_radius=55px, merge_angle=15°
        cur_merge_radius = 55
        cur_merge_angle = 15
        merged_flag = st.session_state.get('merged_processed', False)
        # st.write("壁線を手動で編集・調整します。")
        
        # 壁線手動編集モード
        # st.divider()
        st.subheader("🔧 壁線手動編集")
        
        # モード選択タブ
        edit_mode = st.radio(
            "編集モードを選択:",
            #["線を結合", "線を追加", "線を削除", "窓を追加", "照明を配置", "オブジェクトを配置", "床を追加"],
            ["線を結合", "線を追加", "線を削除", "窓を追加", "オブジェクトを配置"],
            horizontal=True,
            #help="線を結合：2つの壁線を繋ぐ\n\n窓を追加：窓で分断された2本の壁を上下の壁で繋ぐ\n\n線を追加：新しい壁線を追加\n\n線を削除：選択範囲の壁を削除\n\n照明を配置：クリック位置にスポットライトを配置\n\nオブジェクトを配置：キッチンボードなどの家具を配置\n\n床を追加：四角形範囲を選択して床を追加"
            help="線を結合：2つの壁線を繋ぐ\n\n線を追加：新しい壁線を追加\n\n線を削除：選択範囲の壁を削除\n\n窓を追加：窓で分断された2本の壁を上下の壁で繋ぐ\n\nオブジェクトを配置：キッチンボードなどの家具を配置"
        )
        
        if edit_mode == "線を結合":
            st.markdown(
                "結合したい2つの壁線の両端を選択して、壁線を結合します。\n\n"
                "壁全体を囲む必要はなく、結合する2つの壁の両端を囲む四角形を作るように2点クリックしてください。\n\n"
            )
        elif edit_mode == "窓を追加":
            st.markdown(
                "窓追加したい2本の壁の両端を囲むと、その間に窓を追加します。\n\n"
                "壁全体を囲む必要はなく、窓追加したい2つの壁の両端を囲む四角形を作るように2点クリックしてください。\n\n"
                "一条工務店の型番から窓のサイズ（高さと床からの距離）を入力できます。"
            )
        elif edit_mode == "線を追加":
            st.markdown(
                "壁を追加したい範囲を2点クリックすると、壁線が自動生成されます。"
            )
        elif edit_mode == "線を削除":
            st.markdown(
                "削除したい壁線を指定すると、四角形で囲んだ範囲に完全に含まれる全ての壁線が削除されます。"
            )
        elif edit_mode == "オブジェクトを配置":
            st.markdown(
                "オブジェクトを配置したい範囲を2点クリックすると、その範囲に合わせて家具(オブジェクト)を配置します。"
            )
        elif edit_mode == "床を追加":
            st.markdown(
                "床を追加したい範囲を2点クリックすると、その範囲に床を追加します。\n\n"
                "**複数の床を追加可能**で、非四角形の間取りにも対応できます。"
            )
        
        with st.expander("💡 使い方", expanded=False):
            if edit_mode == "線を結合":
                st.markdown(
                    "**複数線結合の手順:**\n\n"
                    "1. 下の画像上で**2回クリック**して四角形の対角を指定\n\n"
                    "2. 結合したい壁線の端点を四角形で囲む\n\n"
                    "3. さらに結合したい箇所があれば手順1-2を繰り返す\n\n"
                    "4. 「🔗 結合実行」で全ての選択範囲を一括結合\n\n"
                )
            elif edit_mode == "窓を追加":
                st.markdown(
                    "**窓追加の手順:**\n\n"
                    "1. 下の画像上で**2回クリック**して窓を追加したい2本の壁の端点を四角形で囲む（端点付近だけでOK）\n\n"
                    "2. さらに窓を追加したければ手順1を繰り返す\n\n"
                    "3. 窓のサイズを入力:\n\n"
                    "   - 窓の高さ（mm）: 例 1200mm\n\n"
                    "   - 床から窓下端までの高さ（mm）: 例 900mm\n\n"
                    "4. 「🪟 窓追加実行」で床側と天井側の壁を追加\n\n"
                    "**特徴:**\n\n"
                    "- 一条工務店の図面表記からそのまま入力可能\n\n"
                )
            elif edit_mode == "線を追加":
                st.markdown(
                    "**線追加の手順:**\n\n"
                    "1. 下の画像上で**2回クリック**して追加したい線の位置を選択する\n\n"
                    "2. さらに線を追加したければ手順1を繰り返す\n\n"
                    "3. 「➕ 線追加実行」で全ての選択範囲に線を追加\n\n"
                )
            elif edit_mode == "線を削除":
                st.markdown(
                    "**線削除の手順:**\n\n"
                    "1. 下の画像上で**2回クリック**して削除したい壁線を四角形で囲む（複数本可）\n\n"
                    "2. さらに削除対象を追加したければ手順1を繰り返す\n\n"
                    "3. 「🗑️ 削除実行」で選択した全ての四角形内の壁を一括削除\n\n"
                )
            elif edit_mode == "オブジェクトを配置":
                st.markdown(
                    "**オブジェクト配置の手順:**\n\n"
                    "1. 下の画像上で**2回クリック**してオブジェクトを配置したい領域を四角形で囲む\n\n"
                    "2. 配置するオブジェクトの種類を選択\n\n"
                    "3. 「🪑 オブジェクト配置実行」で家具を配置\n\n"
                    )
            elif edit_mode == "床を追加":
                st.markdown(
                    "**床追加の手順:**\n\n"
                    "1. 下の画像上で**2回クリック**して四角形の対角を指定\n\n"
                    "2. 床を配置したい範囲を**四角形で囲む**\n\n"
                    "3. 「➕ この選択を追加」ボタンで追加（色が変わります）\n\n"
                    "4. さらに床を追加したければ手順1-3を繰り返す\n\n"
                    "5. 「🟫 床追加実行」で全ての選択範囲に床を追加\n\n"
                    "**特徴:**\n\n"
                    "- **複数の床を配置可能**：L字型やコの字型など複雑な間取りに対応\n\n"
                    "- **自動床の無効化**：床を1つでも追加すると自動生成の大きな床は非表示\n\n"
                    "- 色分け表示：各選択範囲が異なる色で表示される"
                    )
        
        # セッションステートで四角形座標を管理
        if 'rect_coords' not in st.session_state:
            st.session_state.rect_coords = []
        if 'rect_coords_list' not in st.session_state:
            st.session_state.rect_coords_list = []  # 確定した四角形のリスト
        if 'reset_flag' not in st.session_state:
            st.session_state.reset_flag = False
        if 'last_click' not in st.session_state:
            st.session_state.last_click = None
        if 'merge_result' not in st.session_state:
            st.session_state.merge_result = None
        if 'edit_mode_state' not in st.session_state:
            st.session_state.edit_mode_state = "線を結合"  # 現在のモード

        # 窓追加用の入力フォーム（画面上部にまとめて表示）
        if edit_mode == "窓を追加":
            try:
                json_data_tmp = json.loads(st.session_state.json_bytes.decode("utf-8"))
                walls_tmp = json_data_tmp.get('walls', [])
                heights_tmp = [w.get('height', 2.4) for w in walls_tmp if 'height' in w]
                default_room_height = min(max(heights_tmp) if heights_tmp else 2.4, 10.0)
            except Exception:
                default_room_height = 2.4

            # 窓追加モード以外の場合のみ、上部に窓パラメータ入力を表示
            if edit_mode != "窓を追加":
                # 他のモード用の処理（必要に応じて）
                pass
        
        # 四角形の色定義（OpenCV BGRフォーマット）
        RECT_COLORS = [
            (255, 0, 0),      # 赤
            (0, 255, 0),      # 緑
            (0, 0, 255),      # 青
            (255, 255, 0),    # 黄
            (255, 0, 255),    # マゼンタ
            (0, 255, 255),    # シアン
        ]
        
        # 距離閾値（メートル）: 手動結合時に近接した別壁と誤結合しないよう、実用的な閾値を使用
        # 0.3m（30cm）程度に設定。必要に応じて調整してください。
        distance_threshold = 0.3
        
        # 編集結果の表示（セッション状態に保存されている場合）
        if st.session_state.merge_result is not None:
            result = st.session_state.merge_result
            
            st.success("🎉 編集完了！")
            st.markdown("### 📊 編集前後の比較")
            
            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown("**編集前**")
                st.image(Image.open(io.BytesIO(result['original_viz_bytes'])), use_container_width=True)
            with col_after:
                st.markdown("**編集後**")
                st.image(Image.open(io.BytesIO(result['edited_viz_bytes'])), use_container_width=True)
            
            # 統計情報の比較
            st.markdown("### 📈 統計")
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("編集前の壁セグメント数", result['json_data']['metadata']['total_walls'])
            with col_stat2:
                st.metric(
                    "編集後の壁セグメント数", 
                    result['updated_json']['metadata']['total_walls'],
                    delta=result['updated_json']['metadata']['total_walls'] - result['json_data']['metadata']['total_walls']
                )
            
            # デバッグログを表示（rerun後も表示される）
            if 'debug_log' in result and result['debug_log']:
                with st.expander("🔍 デバッグログ（詳細情報）", expanded=True):
                    for log_entry in result['debug_log']:
                        st.text(log_entry)
            
            # セッションステート更新の確認
            st.divider()
            col_save, col_discard = st.columns(2)
            with col_save:
                if st.button("💾 この結果を保存して続行", type="primary"):
                    # JSON・画像を更新
                    st.session_state.json_bytes = result['temp_json_path'].read_bytes()
                    st.session_state.json_name = "walls_3d_edited.json"
                    st.session_state.viz_bytes = result['temp_viz_path'].read_bytes()
                    
                    # 3DビューアHTMLも更新
                    st.session_state.viewer_html_bytes = result['viewer_html_bytes']
                    st.session_state.viewer_html_name = result['temp_viewer_path'].name
                    
                    # 状態を完全にクリア
                    st.session_state.rect_coords = []
                    st.session_state.rect_coords_list = []
                    st.session_state.last_click = None
                    st.session_state.reset_flag = False
                    st.session_state.merge_result = None
                    
                    st.success("✅ 保存しました。さらに編集を続けることができます。")
                    time.sleep(0.5)
                    st.rerun()
            with col_discard:
                if st.button("❌ この結果を破棄"):
                    # 元のJSON・画像を復元
                    original_json_data = result['json_data']
                    original_viz_bytes = result['original_viz_bytes']
                    
                    # JSONを一時ファイルに書き戻す
                    temp_json_path = Path(st.session_state.out_dir) / "walls_3d_edited.json"
                    with open(temp_json_path, 'w', encoding='utf-8') as f:
                        json.dump(original_json_data, f, indent=2, ensure_ascii=False)
                    
                    # セッション状態を元に戻す
                    st.session_state.json_bytes = temp_json_path.read_bytes()
                    st.session_state.viz_bytes = original_viz_bytes
                    
                    # 状態をクリア
                    st.session_state.rect_coords = []
                    st.session_state.rect_coords_list = []
                    st.session_state.last_click = None
                    st.session_state.reset_flag = False
                    st.session_state.merge_result = None
                    st.info("✅ 編集を破棄して元に戻しました。")
                    st.rerun()
        else:
            # 編集結果がない場合のみ、編集UIを表示
            
            # 照明配置モードの場合は独自のUIを表示
            if edit_mode == "照明を配置":
                    # 照明配置モード
                    st.markdown(
                        "可視化画像上で**クリック**してスポットライトを配置します。"
                    )
                    
                    # JSON初期化（初回のみ）
                    if 'lights_list' not in st.session_state:
                        json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                        st.session_state.lights_list = json_data['metadata'].get('lights', [])
                    
                    # 最後のクリック位置を記録（重複防止）
                    if 'last_light_click' not in st.session_state:
                        st.session_state.last_light_click = None
                    
                    # 画像を読み込み
                    viz_image = Image.open(io.BytesIO(st.session_state.viz_bytes))
                    viz_image_resized, scale_ratio, _, _ = _prepare_display_from_pil(viz_image, max_width=DISPLAY_IMAGE_WIDTH)
                    
                    # 画像表示とクリック処理
                    click_result = streamlit_image_coordinates(
                        np.array(viz_image_resized),
                        key=f"light_placement_{st.session_state.viz_name}"
                    )
                    
                    if click_result is not None:
                        # クリック座標を取得
                        click_x = click_result.get('x')
                        click_y = click_result.get('y')
                        
                        if click_x is not None and click_y is not None:
                            # 表示座標を元画像座標とメートル座標に変換
                            json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            walls = json_data['walls']
                            all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                            all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                            min_x, min_y = min(all_x), min(all_y)
                            scale = int(st.session_state.viz_scale)
                            margin = 50

                            meter_x, meter_y, actual_x, actual_y = _display_to_meter(
                                click_x, click_y, scale_ratio, viz_image.height, margin, scale, min_x, min_y
                            )

                            # デバッグ情報
                            st.write(f"**デバッグ**: クリック位置(px)=({click_x}, {click_y}) → 実座標(px)=({actual_x}, {actual_y})")
                            st.write(f"**デバッグ**: 画像サイズ={viz_image.width}x{viz_image.height}px, scale={scale}px/m, margin={margin}px")
                            st.write(f"**デバッグ**: 座標範囲: X=[{min_x:.2f}, {max(all_x):.2f}]m, Y=[{min_y:.2f}, {max(all_y):.2f}]m")
                            st.write(f"**デバッグ**: 変換後のメートル座標: X={meter_x:.3f}m, Y={meter_y:.3f}m")
                            
                            # 重複クリック防止（同じ座標のクリックを無視）
                            current_click = (click_x, click_y)
                            if st.session_state.last_light_click != current_click:
                                # 照明を追加（高さは天井高さの80%）
                                avg_wall_height = sum(w['height'] for w in walls) / len(walls) if walls else 2.7
                                light_z = avg_wall_height * 0.8
                                
                                new_light = {
                                    'position': [round(meter_x, 3), round(light_z, 3), round(meter_y, 3)],
                                    'intensity': 1.0,
                                    'color': '#ffffff'
                                }
                                st.session_state.lights_list.append(new_light)
                                st.session_state.last_light_click = current_click
                                st.success(f"✅ 照明を追加しました (位置: [X={meter_x:.3f}m, Z={light_z:.3f}m, Y={meter_y:.3f}m])")
                                st.rerun()
                    
                    # 照明リスト表示
                    st.markdown("### 配置済み照明")
                    if st.session_state.lights_list:
                        col_clear, col_count = st.columns([1, 3])
                        with col_clear:
                            if st.button("🗑️ 全削除"):
                                st.session_state.lights_list = []
                                st.session_state.last_light_click = None
                                st.rerun()
                        with col_count:
                            st.write(f"合計: {len(st.session_state.lights_list)} 個")
                        
                        for i, light in enumerate(st.session_state.lights_list):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                pos = light['position']
                                st.write(f"照明 {i+1}: [X={pos[0]:.3f}m, Y(高)={pos[1]:.3f}m, Z={pos[2]:.3f}m]")
                            with col2:
                                if st.button("削除", key=f"delete_light_{i}"):
                                    st.session_state.lights_list.pop(i)
                                    st.rerun()
                    else:
                        st.info("照明はまだ配置されていません")
                    
                    # 保存ボタン
                    if st.button("💾 照明設定を保存", type="primary"):
                        with st.spinner("照明設定を保存中..."):
                            try:
                                st.write("🔧 デバッグ: 保存処理開始")
                                json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                json_data['metadata']['lights'] = st.session_state.lights_list
                                
                                st.write(f"🔧 デバッグ: 照明数={len(st.session_state.lights_list)}")
                                
                                out_dir = Path(st.session_state.out_dir)
                                json_path = out_dir / st.session_state.json_name
                                
                                st.write(f"🔧 デバッグ: JSONパス={json_path}")
                                
                                with open(json_path, 'w', encoding='utf-8') as f:
                                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                                
                                st.write("🔧 デバッグ: JSON保存完了")
                                
                                # 照明付き3DビューアHTMLを別名で生成
                                viewer_html_lights = out_dir / "viewer_3d_with_lights.html"
                                
                                st.write(f"🔧 デバッグ: HTMLパス={viewer_html_lights}")
                                
                                _generate_3d_viewer_html(json_path, viewer_html_lights, with_lights=True)
                                
                                st.write("🔧 デバッグ: HTML生成完了")
                                
                                # セッション状態を更新
                                st.session_state.json_bytes = json_path.read_bytes()
                                st.session_state.viewer_html_lights_bytes = viewer_html_lights.read_bytes()
                                st.session_state.viewer_html_lights_name = viewer_html_lights.name
                                
                                st.success("✅ 照明設定を保存しました")
                                st.info(f"📊 照明付き3Dビューア: {viewer_html_lights.name} を生成しました")
                                st.write("🔄 画面を更新します...")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ エラー: {e}")
                                import traceback
                                st.code(traceback.format_exc())
            else:
                # 可視化画像を読み込み（照明配置モード以外）
                if st.session_state.viz_bytes and edit_mode != "照明を配置":
                    viz_img = Image.open(io.BytesIO(st.session_state.viz_bytes))
                    
                    # 選択範囲を描画した画像を作成
                    import cv2
                    display_img_array = np.array(viz_img.copy())
                    
                    # リセットボタンと追加ボタン（画像の前に配置）
                    col_reset, col_add, col_exec = st.columns(3)
                    with col_reset:
                        if st.button("🗑️ 選択リセット"):
                            st.session_state.rect_coords = []
                            st.session_state.rect_coords_list = []
                            st.session_state.reset_flag = True
                            st.session_state.last_click = None
                            st.session_state.merge_result = None  # 結合結果もクリア
                            st.rerun()
                    
                    # 確定済みの四角形を描画（異なる色で）
                    for idx, (p1, p2) in enumerate(st.session_state.rect_coords_list):
                        color = RECT_COLORS[idx % len(RECT_COLORS)]
                        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                        
                        # 線削除モードの場合は四角形内の壁を色変更
                        if edit_mode == "線を削除":
                            try:
                                json_data_del = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                walls_del = json_data_del.get('walls', [])
                                
                                all_x_del = [w['start'][0] for w in walls_del] + [w['end'][0] for w in walls_del]
                                all_y_del = [w['start'][1] for w in walls_del] + [w['end'][1] for w in walls_del]
                                min_x_del = min(all_x_del)
                                min_y_del = min(all_y_del)
                                max_x_del = max(all_x_del)
                                max_y_del = max(all_y_del)
                                
                                scale_del = int(st.session_state.viz_scale)
                                margin_del = 50
                                img_height_del = viz_img.height
                                
                                rect_del = {
                                    'left': x1,
                                    'top': y1,
                                    'width': x2 - x1,
                                    'height': y2 - y1
                                }
                                
                                walls_in_rect_del = _filter_walls_strictly_in_rect(
                                    walls_del, rect_del, scale_del, margin_del,
                                    img_height_del, min_x_del, min_y_del, max_x_del, max_y_del
                                )
                                
                                # 削除対象の壁を赤色で太く描画
                                if len(walls_in_rect_del) > 0:
                                    for wall in walls_in_rect_del:
                                        start_m = wall['start']
                                        end_m = wall['end']
                                        
                                        # メートル→ピクセル変換
                                        start_px_x = int((start_m[0] - min_x_del) * scale_del) + margin_del
                                        start_px_y = img_height_del - (int((start_m[1] - min_y_del) * scale_del) + margin_del)
                                        end_px_x = int((end_m[0] - min_x_del) * scale_del) + margin_del
                                        end_px_y = img_height_del - (int((end_m[1] - min_y_del) * scale_del) + margin_del)
                                        
                                        # 赤色で太い線を描画（削除対象）
                                        cv2.line(display_img_array, (start_px_x, start_px_y), (end_px_x, end_px_y), (0, 0, 255), 8)
                                else:
                                    # 壁が見つからない場合は四角形を表示
                                    overlay = display_img_array.copy()
                                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                                    cv2.addWeighted(overlay, 0.25, display_img_array, 0.75, 0, display_img_array)
                                    cv2.rectangle(display_img_array, (x1, y1), (x2, y2), color, 3)
                                    cv2.putText(display_img_array, f"{idx+1}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            except Exception:
                                # エラー時は四角形を表示
                                overlay = display_img_array.copy()
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                                cv2.addWeighted(overlay, 0.25, display_img_array, 0.75, 0, display_img_array)
                                cv2.rectangle(display_img_array, (x1, y1), (x2, y2), color, 3)
                                cv2.putText(display_img_array, f"{idx+1}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        # 窓追加モードまたは線を結合モードの場合は四角形ではなく追加予定の壁を線で表示
                        elif edit_mode == "窓を追加" or edit_mode == "線を結合":
                            try:
                                json_data_confirmed = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                walls_confirmed = json_data_confirmed.get('walls', [])
                                
                                all_x_confirmed = [w['start'][0] for w in walls_confirmed] + [w['end'][0] for w in walls_confirmed]
                                all_y_confirmed = [w['start'][1] for w in walls_confirmed] + [w['end'][1] for w in walls_confirmed]
                                min_x_confirmed = min(all_x_confirmed)
                                min_y_confirmed = min(all_y_confirmed)
                                max_x_confirmed = max(all_x_confirmed)
                                max_y_confirmed = max(all_y_confirmed)
                                
                                scale_confirmed = int(st.session_state.viz_scale)
                                margin_confirmed = 50
                                img_height_confirmed = viz_img.height
                                
                                rect_confirmed = {
                                    'left': x1,
                                    'top': y1,
                                    'width': x2 - x1,
                                    'height': y2 - y1
                                }
                                
                                # 窓追加モードおよび線を結合モードでは端点/交差ベースで検出（端点だけ囲む操作に対応）
                                if edit_mode in ("窓を追加", "線を結合"):
                                    walls_in_rect_confirmed = _filter_walls_by_endpoints_in_rect(
                                        walls_confirmed, rect_confirmed, scale_confirmed, margin_confirmed,
                                        img_height_confirmed, min_x_confirmed, min_y_confirmed, max_x_confirmed, max_y_confirmed,
                                        tolerance=0, debug=False
                                    )
                                else:
                                    walls_in_rect_confirmed = _filter_walls_strictly_in_rect(
                                        walls_confirmed, rect_confirmed, scale_confirmed, margin_confirmed,
                                        img_height_confirmed, min_x_confirmed, min_y_confirmed, max_x_confirmed, max_y_confirmed
                                    )

                                # プレビューと同じく、端点が重なって余分な線が含まれる場合は縦横判定して最適な2本を選択
                                try:
                                    if len(walls_in_rect_confirmed) >= 3:
                                        # 3本以上：縦横を分類して最も近い平行な壁のペアを選ぶ
                                        best_pair = _select_best_wall_pair_from_4(walls_in_rect_confirmed)
                                        walls_in_rect_confirmed = best_pair if best_pair else walls_in_rect_confirmed[:2]
                                    else:
                                        # 2本以下の場合はそのまま使用
                                        walls_in_rect_confirmed = walls_in_rect_confirmed
                                except Exception:
                                    pass

                                
                                if len(walls_in_rect_confirmed) == 2:
                                    params = st.session_state.get('window_execution_params', {})
                                    window_height_confirmed = params.get('window_height', 1.2)
                                    base_height_confirmed = params.get('base_height', 0.9)
                                    
                                    wall1_confirmed = walls_in_rect_confirmed[0]
                                    wall2_confirmed = walls_in_rect_confirmed[1]
                                    
                                    endpoints_confirmed = [
                                        (wall1_confirmed['start'], wall2_confirmed['start']),
                                        (wall1_confirmed['start'], wall2_confirmed['end']),
                                        (wall1_confirmed['end'], wall2_confirmed['start']),
                                        (wall1_confirmed['end'], wall2_confirmed['end']),
                                    ]
                                    
                                    min_dist_confirmed = float('inf')
                                    window_start_confirmed = None
                                    window_end_confirmed = None
                                    
                                    for p1_win, p2_win in endpoints_confirmed:
                                        dist_win = math.sqrt((p1_win[0] - p2_win[0])**2 + (p1_win[1] - p2_win[1])**2)
                                        if dist_win < min_dist_confirmed:
                                            min_dist_confirmed = dist_win
                                            window_start_confirmed = p1_win
                                            window_end_confirmed = p2_win
                                    
                                    if window_start_confirmed and window_end_confirmed:
                                        start_px = [
                                            int((window_start_confirmed[0] - min_x_confirmed) * scale_confirmed) + margin_confirmed,
                                            img_height_confirmed - (int((window_start_confirmed[1] - min_y_confirmed) * scale_confirmed) + margin_confirmed)
                                        ]
                                        end_px = [
                                            int((window_end_confirmed[0] - min_x_confirmed) * scale_confirmed) + margin_confirmed,
                                            img_height_confirmed - (int((window_end_confirmed[1] - min_y_confirmed) * scale_confirmed) + margin_confirmed)
                                        ]
                                        
                                        # 線を結合モードは青色、窓追加モードは赤色
                                        line_color = (255, 0, 0) if edit_mode == "線を結合" else (0, 0, 255)  # BGR形式
                                        cv2.line(display_img_array, tuple(start_px), tuple(end_px), line_color, 5)
                                        cv2.circle(display_img_array, tuple(start_px), 8, line_color, -1)
                                        cv2.circle(display_img_array, tuple(end_px), 8, line_color, -1)
                                        
                                        # 番号を描画（線の上側、白背景で視認性向上）
                                        mid_x = (start_px[0] + end_px[0]) // 2
                                        mid_y = (start_px[1] + end_px[1]) // 2
                                        text = f"{idx+1}"
                                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                                        text_x = mid_x - text_size[0] // 2
                                        text_y = mid_y - 20  # 線の上側に配置
                                        # 白背景の四角形を描画
                                        cv2.rectangle(display_img_array, 
                                                    (text_x - 5, text_y - text_size[1] - 5),
                                                    (text_x + text_size[0] + 5, text_y + 5),
                                                    (255, 255, 255), -1)
                                        # 黒枠を描画
                                        cv2.rectangle(display_img_array, 
                                                    (text_x - 5, text_y - text_size[1] - 5),
                                                    (text_x + text_size[0] + 5, text_y + 5),
                                                    (0, 0, 0), 2)
                                        # 番号を描画（黒文字）
                                        cv2.putText(display_img_array, text, (text_x, text_y), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                                else:
                                    # 2本検出できない場合は四角形を表示
                                    overlay = display_img_array.copy()
                                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                                    cv2.addWeighted(overlay, 0.25, display_img_array, 0.75, 0, display_img_array)
                                    cv2.rectangle(display_img_array, (x1, y1), (x2, y2), color, 3)
                                    cv2.putText(display_img_array, f"{idx+1}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            except Exception:
                                # エラー時は通常の四角形表示
                                overlay = display_img_array.copy()
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                                cv2.addWeighted(overlay, 0.25, display_img_array, 0.75, 0, display_img_array)
                                cv2.rectangle(display_img_array, (x1, y1), (x2, y2), color, 3)
                                cv2.putText(display_img_array, f"{idx+1}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        else:
                            # 窓追加モード・線を結合モード以外は通常の四角形表示
                            # 半透明の四角形を描画
                            overlay = display_img_array.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                            cv2.addWeighted(overlay, 0.25, display_img_array, 0.75, 0, display_img_array)
                            # 四角形の枠線を描画
                            cv2.rectangle(display_img_array, (x1, y1), (x2, y2), color, 3)
                            # 番号を描画
                            cv2.putText(display_img_array, f"{idx+1}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # 現在選択中の四角形を描画（赤色で表示）
                    if len(st.session_state.rect_coords) == 1:
                        # 1点目を赤い円で表示
                        red_color = (0, 0, 255)  # BGR形式で赤色
                        cv2.circle(display_img_array, st.session_state.rect_coords[0], 12, red_color, -1)  # 塗りつぶし円
                        cv2.circle(display_img_array, st.session_state.rect_coords[0], 15, (255, 255, 255), 2)  # 白枠
                    elif len(st.session_state.rect_coords) == 2:
                        # 2点を赤い円で表示し、四角形も描画
                        red_color = (0, 0, 255)  # BGR形式で赤色
                        p1, p2 = st.session_state.rect_coords
                        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                        # 半透明の四角形を描画
                        overlay = display_img_array.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), red_color, -1)
                        cv2.addWeighted(overlay, 0.3, display_img_array, 0.7, 0, display_img_array)
                        # 四角形の枠線を描画
                        cv2.rectangle(display_img_array, (x1, y1), (x2, y2), red_color, 3)
                        # 両端に赤い円を表示
                        cv2.circle(display_img_array, p1, 12, red_color, -1)
                        cv2.circle(display_img_array, p1, 15, (255, 255, 255), 2)
                        cv2.circle(display_img_array, p2, 12, red_color, -1)
                        cv2.circle(display_img_array, p2, 15, (255, 255, 255), 2)
                        
                        # 窓追加モードまたは線を結合モードの場合、追加予定の壁を線で表示
                        if edit_mode == "窓を追加" or edit_mode == "線を結合":
                            try:
                                json_data_preview = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                walls_preview = json_data_preview.get('walls', [])
                                
                                # 可視化画像のパラメータを取得
                                all_x_preview = [w['start'][0] for w in walls_preview] + [w['end'][0] for w in walls_preview]
                                all_y_preview = [w['start'][1] for w in walls_preview] + [w['end'][1] for w in walls_preview]
                                min_x_preview = min(all_x_preview)
                                min_y_preview = min(all_y_preview)
                                max_x_preview = max(all_x_preview)
                                max_y_preview = max(all_y_preview)
                                
                                scale_preview = int(st.session_state.viz_scale)
                                margin_preview = 50
                                img_height_preview = viz_img.height
                                
                                rect_preview = {
                                    'left': x1,
                                    'top': y1,
                                    'width': x2 - x1,
                                    'height': y2 - y1
                                }
                                
                                # 範囲内の壁を検出（窓追加/線を結合では端点/交差ベース）
                                if edit_mode in ("窓を追加", "線を結合"):
                                    walls_in_rect_preview = _filter_walls_by_endpoints_in_rect(
                                        walls_preview, rect_preview, scale_preview, margin_preview,
                                        img_height_preview, min_x_preview, min_y_preview, max_x_preview, max_y_preview,
                                        tolerance=0, debug=False
                                    )
                                else:
                                    walls_in_rect_preview = _filter_walls_strictly_in_rect(
                                        walls_preview, rect_preview, scale_preview, margin_preview, 
                                        img_height_preview, min_x_preview, min_y_preview, max_x_preview, max_y_preview
                                    )

                                # 追加表示用に、縦横を判定して最適な線を選択（プレビュー整合性）
                                try:
                                    if len(walls_in_rect_preview) >= 3:
                                        # 3本以上：縦横を分類して最も近い平行な壁のペアを選ぶ
                                        best_pair = _select_best_wall_pair_from_4(walls_in_rect_preview)
                                        walls_in_rect_filtered = best_pair if best_pair else walls_in_rect_preview[:2]
                                    else:
                                        # 2本以下の場合はそのまま使用
                                        walls_in_rect_filtered = walls_in_rect_preview
                                except Exception:
                                    walls_in_rect_filtered = walls_in_rect_preview

                                # プレビュー時の詳細デバッグ出力（2点目選択直後に表示）
                                try:
                                    # プレビュー時の内部マッピングは表示しない（ログのみ残す）
                                    preview_debug = []
                                    for w in walls_preview:
                                        x1w = int((w['start'][0] - min_x_preview) * scale_preview) + margin_preview
                                        y1w = img_height_preview - (int((w['start'][1] - min_y_preview) * scale_preview) + margin_preview)
                                        x2w = int((w['end'][0] - min_x_preview) * scale_preview) + margin_preview
                                        y2w = img_height_preview - (int((w['end'][1] - min_y_preview) * scale_preview) + margin_preview)
                                        in_rect = (
                                            (rect_preview['left'] <= x1w <= rect_preview['left'] + rect_preview['width'] and rect_preview['top'] <= y1w <= rect_preview['top'] + rect_preview['height']) or
                                            (rect_preview['left'] <= x2w <= rect_preview['left'] + rect_preview['width'] and rect_preview['top'] <= y2w <= rect_preview['top'] + rect_preview['height']) or
                                            _line_intersects_rect(x1w, y1w, x2w, y2w, rect_preview, tolerance=20)
                                        )
                                        preview_debug.append({'id': w.get('id'), 'start_px': (x1w, y1w), 'end_px': (x2w, y2w), 'in_rect': in_rect})
                                    try:
                                        append_debug(f"Preview mapping: detected={[d['id'] for d in preview_debug if d.get('in_rect')]}, total_checked={len(preview_debug)}")
                                    except Exception:
                                        pass
                                except Exception:
                                    pass

                                # 検出が2本以上ある場合、追加予定の線を描画
                                if len(walls_in_rect_filtered) >= 2:
                                    # まず窓パラメータを取得（表示用）
                                    params = st.session_state.get('window_execution_params', {})
                                    window_height_preview = params.get('window_height', 1.2)
                                    base_height_preview = params.get('base_height', 0.9)
                                    room_height_preview = params.get('room_height', 2.4)

                                    # 2本より多い場合は四角形の主方向に沿って最も離れた2本を選ぶ
                                    selected_walls = None
                                    try:
                                        rect_w = rect_preview.get('width', 0)
                                        rect_h = rect_preview.get('height', 0)
                                        if len(walls_in_rect_filtered) == 2:
                                            selected_walls = [walls_in_rect_filtered[0], walls_in_rect_filtered[1]]
                                        else:
                                            if rect_w > rect_h:
                                                walls_by_x = sorted(walls_in_rect_filtered, key=lambda w: min(w['start'][0], w['end'][0]))
                                                selected_walls = [walls_by_x[0], walls_by_x[-1]]
                                            else:
                                                walls_by_y = sorted(walls_in_rect_filtered, key=lambda w: min(w['start'][1], w['end'][1]))
                                                selected_walls = [walls_by_y[0], walls_by_y[-1]]
                                    except Exception:
                                        selected_walls = walls_in_rect_filtered[:2]

                                    # 選ばれた2本から最短端点対を探す
                                    wall1_preview = selected_walls[0]
                                    wall2_preview = selected_walls[1]
                                    endpoints_preview = [
                                        (wall1_preview['start'], wall2_preview['start']),
                                        (wall1_preview['start'], wall2_preview['end']),
                                        (wall1_preview['end'], wall2_preview['start']),
                                        (wall1_preview['end'], wall2_preview['end']),
                                    ]

                                    min_dist_preview = float('inf')
                                    window_start_preview = None
                                    window_end_preview = None
                                    for p1_win, p2_win in endpoints_preview:
                                        dist_win = math.sqrt((p1_win[0] - p2_win[0])**2 + (p1_win[1] - p2_win[1])**2)
                                        if dist_win < min_dist_preview:
                                            min_dist_preview = dist_win
                                            window_start_preview = p1_win
                                            window_end_preview = p2_win

                                    # メートル座標をピクセル座標に変換して線を描画
                                    if window_start_preview and window_end_preview:
                                        start_px = [
                                            int((window_start_preview[0] - min_x_preview) * scale_preview) + margin_preview,
                                            img_height_preview - (int((window_start_preview[1] - min_y_preview) * scale_preview) + margin_preview)
                                        ]
                                        end_px = [
                                            int((window_end_preview[0] - min_x_preview) * scale_preview) + margin_preview,
                                            img_height_preview - (int((window_end_preview[1] - min_y_preview) * scale_preview) + margin_preview)
                                        ]

                                        # 線を結合モードは青色、窓追加モードは赤色で太い線を描画
                                        line_color = (255, 0, 0) if edit_mode == "線を結合" else (0, 0, 255)  # BGR形式
                                        cv2.line(display_img_array, tuple(start_px), tuple(end_px), line_color, 5)
                                        cv2.circle(display_img_array, tuple(start_px), 8, line_color, -1)
                                        cv2.circle(display_img_array, tuple(end_px), 8, line_color, -1)

                                        # プレビューで選択したペアをセッションに保存（実行時に優先利用）
                                        try:
                                            st.session_state['last_preview_pair'] = [wall1_preview.get('id'), wall2_preview.get('id')]
                                            st.session_state['last_preview_rect'] = rect_preview
                                            # フィルタ済みの検出IDリストも保存（デバッグ/実行整合用）
                                            try:
                                                st.session_state['last_preview_detected_ids'] = [w.get('id') for w in walls_in_rect_preview]
                                                st.session_state['last_preview_filtered_ids'] = [w.get('id') for w in walls_in_rect_filtered]
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass

                                        # プレビュー用に番号も描画（確定リストに追加されていない未追加プレビュー）
                                        try:
                                            preview_idx = len(st.session_state.get('rect_coords_list', [])) + 1
                                            mid_x = (start_px[0] + end_px[0]) // 2
                                            mid_y = (start_px[1] + end_px[1]) // 2
                                            text = f"{preview_idx}"
                                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                                            text_x = mid_x - text_size[0] // 2
                                            text_y = mid_y - 20
                                            cv2.rectangle(display_img_array,
                                                          (text_x - 5, text_y - text_size[1] - 5),
                                                          (text_x + text_size[0] + 5, text_y + 5),
                                                          (255, 255, 255), -1)
                                            cv2.rectangle(display_img_array,
                                                          (text_x - 5, text_y - text_size[1] - 5),
                                                          (text_x + text_size[0] + 5, text_y + 5),
                                                          (0, 0, 0), 2)
                                            cv2.putText(display_img_array, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                                        except Exception:
                                            pass
                            except Exception as e:
                                # プレビュー描画のエラーは無視（通常の四角形表示を続行）
                                pass
                    
                    display_img = Image.fromarray(display_img_array)
                    display_img_resized, scale_ratio, _, _ = _prepare_display_from_pil(display_img, max_width=DISPLAY_IMAGE_WIDTH)
                    
                    # UI表示：モード別
                    if edit_mode == "線を削除":
                        # 削除モード：2点選択で四角形を指定
                        if len(st.session_state.rect_coords) == 1:
                            pass
                            #st.info(f"✓ 1点目選択: ({st.session_state.rect_coords[0][0]}, {st.session_state.rect_coords[0][1]})")
                        elif len(st.session_state.rect_coords) == 2:
                            p1, p2 = st.session_state.rect_coords
                            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                            st.success(f"✅ 2点選択完了: ({x1}, {y1}) - ({x2}, {y2})")
                        st.write("画像を2点クリックして対象の壁を選択してください（1点目→2点目）")
                    elif edit_mode == "スケール校正":
                        # スケール校正モード：2点選択で線を囲む
                        if len(st.session_state.rect_coords) == 1:
                            pass
                            #st.info(f"✓ 1点目選択: ({st.session_state.rect_coords[0][0]}, {st.session_state.rect_coords[0][1]})")
                        elif len(st.session_state.rect_coords) == 2:
                            p1, p2 = st.session_state.rect_coords
                            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                            px_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            #st.success(f"✅ 2点選択完了: ({x1}, {y1}) - ({x2}, {y2})\n\n線の長さ: {px_distance:.1f}px")
                        st.write("画像をクリックして四角形の2点を指定してください（1点目→2点目）")
                    else:
                        # 結合・追加モード：2点選択
                        if len(st.session_state.rect_coords) == 1:
                            pass
                            #st.info(f"✓ 1点目選択: ({st.session_state.rect_coords[0][0]}, {st.session_state.rect_coords[0][1]})")
                        elif len(st.session_state.rect_coords) == 2:
                            # 窓追加モードで自動追加される場合は、前のrerunでrect_coordsがクリアされるため、
                            # このブロックに到達しない。失敗時のみここに到達する
                            if edit_mode != "窓を追加":
                                p1, p2 = st.session_state.rect_coords
                                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                                color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][len(st.session_state.rect_coords_list) % 6]
                                #st.success(f"✅ 2点選択完了（{color_name}）: ({x1}, {y1}) - ({x2}, {y2})")
                        st.write("画像をクリックして四角形の2点を指定してください（1点目→2点目）")
                        
                        # 窓追加モードで2点選択完了時：壁検出結果をハイライト表示
                        if edit_mode in ("窓を追加", "線を結合") and len(st.session_state.rect_coords) == 2:
                            try:
                                json_data_check = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                walls_check = json_data_check['walls']
                                
                                all_x_check = [w['start'][0] for w in walls_check] + [w['end'][0] for w in walls_check]
                                all_y_check = [w['start'][1] for w in walls_check] + [w['end'][1] for w in walls_check]
                                min_x_check, max_x_check = min(all_x_check), max(all_x_check)
                                min_y_check, max_y_check = min(all_y_check), max(all_y_check)
                                
                                scale_check = int(viz_scale)
                                margin_check = 50
                                img_height_check = viz_img.height
                                
                                p1_check, p2_check = st.session_state.rect_coords
                                x1_check, y1_check = min(p1_check[0], p2_check[0]), min(p1_check[1], p2_check[1])
                                x2_check, y2_check = max(p1_check[0], p2_check[0]), max(p1_check[1], p2_check[1])
                                
                                rect_check = {
                                    'left': x1_check,
                                    'top': y1_check,
                                    'width': x2_check - x1_check,
                                    'height': y2_check - y1_check
                                }
                                
                                # 端点のみを囲む操作に対応するため、厳密フィルタではなく端点/交差ベースのフィルタを使用
                                walls_in_rect_check = _filter_walls_by_endpoints_in_rect(
                                    walls_check, rect_check, scale_check, margin_check,
                                    img_height_check, min_x_check, min_y_check, max_x_check, max_y_check,
                                    tolerance=0, debug=False
                                )

                                # 端点が重なって複数の壁が検出される場合、縦横を判定して最適な2本を選択
                                try:
                                    # デバッグ: 検出された全ての壁の情報を表示
                                    try:
                                        all_wall_details = []
                                        for w in walls_in_rect_check:
                                            dx = abs(w['end'][0] - w['start'][0])
                                            dy = abs(w['end'][1] - w['start'][1])
                                            direction = "縦" if dx < dy else "横"
                                            all_wall_details.append(f"ID{w['id']}({direction})")
                                        append_debug(f"Detected walls before filtering: {', '.join(all_wall_details)}")
                                    except:
                                        pass
                                    
                                    if len(walls_in_rect_check) >= 3:
                                        # 3本以上：縦横を分類して最も近い平行な壁のペアを選ぶ
                                        best_pair = _select_best_wall_pair_from_4(walls_in_rect_check)
                                        walls_in_rect_filtered = best_pair if best_pair else walls_in_rect_check[:2]
                                    else:
                                        # 2本以下の場合はそのまま使用
                                        walls_in_rect_filtered = walls_in_rect_check
                                except Exception:
                                    walls_in_rect_filtered = walls_in_rect_check



                                if len(walls_in_rect_filtered) in (2, 3):
                                    # プレビューで2本検出された場合、確定集合にも反映して表示を整合させる
                                    try:
                                        walls_in_rect_confirmed = walls_in_rect_filtered
                                        # プレビューで2本検出された四角形情報とIDをセッションに保存
                                        try:
                                            st.session_state['last_preview_pair'] = [walls_in_rect_filtered[0]['id'], walls_in_rect_filtered[1]['id']]
                                            st.session_state['last_preview_rect'] = rect_preview
                                            # 検出/フィルタ済みIDもセッション保存して実行時に優先できるようにする
                                            try:
                                                st.session_state['last_preview_detected_ids'] = [w.get('id') for w in walls_in_rect_check]
                                                st.session_state['last_preview_filtered_ids'] = [w.get('id') for w in walls_in_rect_filtered]
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                                    # プレビュー確定表示は不要になったためUI非表示（内部データは保持）
                                    try:
                                        preview_debug_confirm = []
                                        for w in walls_check:
                                            x1w = int((w['start'][0] - min_x_check) * scale_check) + margin_check
                                            y1w = img_height_check - (int((w['start'][1] - min_y_check) * scale_check) + margin_check)
                                            x2w = int((w['end'][0] - min_x_check) * scale_check) + margin_check
                                            y2w = img_height_check - (int((w['end'][1] - min_y_check) * scale_check) + margin_check)
                                            in_rect = (
                                                (rect_check['left'] <= x1w <= rect_check['left'] + rect_check['width'] and rect_check['top'] <= y1w <= rect_check['top'] + rect_check['height']) or
                                                (rect_check['left'] <= x2w <= rect_check['left'] + rect_check['width'] and rect_check['top'] <= y2w <= rect_check['top'] + rect_check['height']) or
                                                _line_intersects_rect(x1w, y1w, x2w, y2w, rect_check, tolerance=20)
                                            )
                                            preview_debug_confirm.append({'id': w.get('id'), 'start_px': (x1w, y1w), 'end_px': (x2w, y2w), 'in_rect': in_rect})
                                        try:
                                            append_debug(f"Preview confirmed mapping: detected={[d['id'] for d in preview_debug_confirm if d.get('in_rect')]}, total_checked={len(preview_debug_confirm)}")
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                                    # デバッグ: 選択された壁の詳細情報を表示
                                    try:
                                        wall_details = []
                                        for w in walls_in_rect_filtered:
                                            dx = abs(w['end'][0] - w['start'][0])
                                            dy = abs(w['end'][1] - w['start'][1])
                                            direction = "縦" if dx < dy else "横"
                                            wall_details.append(f"ID{w['id']}({direction}, dx={dx:.2f}, dy={dy:.2f})")
                                        st.info(f"🎯 この範囲に2本の壁が検出されました\n検出数: {len(walls_in_rect_check)}本 → フィルタ後: {len(walls_in_rect_filtered)}本\n選択された壁: {', '.join(wall_details)}")
                                    except Exception:
                                        st.info(f"🎯 この範囲に2本の壁が検出されました（ID: {walls_in_rect_filtered[0]['id']}, {walls_in_rect_filtered[1]['id']}）\n検出数: {len(walls_in_rect_check)}本 → フィルタ後: {len(walls_in_rect_filtered)}本")
                                elif len(walls_in_rect_filtered) == 0:
                                    st.error("❌ **この範囲に壁が検出されませんでした。**\n\n💡 **窓で分断された2本の壁を両方含むように**、もう少し広い範囲を選択してください。")
                                elif len(walls_in_rect_filtered) == 1:
                                    st.warning(f"⚠️ **この範囲に1本の壁しか検出されません。**\n\n💡 **窓で分断された2本の壁を両方含むように**選択してください。\n\n窓の両側（上下または左右）にある壁が2本とも範囲内に入るように、選択範囲を広げてください。")
                                else:
                                    st.warning(f"⚠️ **この範囲に{len(walls_in_rect_filtered)}本の壁が検出されました。**\n\n💡 選択範囲を狭めて余分な壁が含まれないように調整してください。")
                            except Exception:
                                pass
                    
                    # クリック可能な画像を表示（キーを動的に変更して値をリセット）
                    coord_key = f"image_coords_{len(st.session_state.rect_coords_list)}_{len(st.session_state.rect_coords)}"
                    
                    st.markdown(
                        """
                        <p style="font-size: 12px; color: #666; margin-bottom: 8px;">
                        💡 <b>注:</b> 画像が見切れる場合は、ブラウザの画面スケール（Ctrl/Cmd + マイナスキー）を小さくしてください。
                        </p>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # 画像を元のサイズで表示（リサイズなし）
                    value = streamlit_image_coordinates(
                        display_img_resized,
                        key=coord_key
                    )
                    
                    # リサイズ時の座標変換
                    if value is not None and value.get("x") is not None and scale_ratio != 1.0:
                        # 元の座標に変換
                        ox, oy = _display_to_original(value["x"], value["y"], scale_ratio)
                        value["x"] = ox
                        value["y"] = oy

                    # デバッグ: クリック座標を表示
                    #if value is not None and value.get("x") is not None:
                    #    st.caption(
                    #        f"クリック座標: raw=({value['x']}, {value['y']}) | "
                    #        f"表示画像サイズ={display_img_resized.width}x{display_img_resized.height}px | "
                    #        f"scale_ratio={scale_ratio:.3f}"
                    #    )
                
                    # クリックされた座標を記録（重複チェック）
                    if value is not None and value.get("x") is not None:
                        new_point = (value["x"], value["y"])
                    
                        if edit_mode == "線を削除":
                            # 削除モード：2点選択で四角形、2点目で自動追加
                            if len(st.session_state.rect_coords) < 2:
                                if len(st.session_state.rect_coords) == 0 or st.session_state.last_click != new_point:
                                    st.session_state.rect_coords.append(new_point)
                                    st.session_state.last_click = new_point
                                    
                                    # 2点目クリック時に自動追加
                                    if len(st.session_state.rect_coords) == 2:
                                        st.session_state.rect_coords_list.append(tuple(st.session_state.rect_coords))
                                        st.session_state.rect_coords = []
                                        st.session_state.last_click = None
                                    
                                    st.rerun()
                        else:
                            # 結合・追加モード：2点選択
                            if len(st.session_state.rect_coords) < 2:
                                if len(st.session_state.rect_coords) == 0 or st.session_state.last_click != new_point:
                                    st.session_state.rect_coords.append(new_point)
                                    st.session_state.last_click = new_point
                                    
                                    # 窓追加モード、線を結合モード、線を追加モード、またはオブジェクト配置モードで2点目クリック時：
                                    # 2本の壁が検出されたら自動追加（オブジェクト配置では四角形をそのまま追加）
                                    if (edit_mode in ("窓を追加", "線を結合", "線を追加", "オブジェクトを配置")) and len(st.session_state.rect_coords) == 2:
                                        try:
                                            json_data_auto = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                            walls_auto = json_data_auto['walls']
                                            
                                            all_x_auto = [w['start'][0] for w in walls_auto] + [w['end'][0] for w in walls_auto]
                                            all_y_auto = [w['start'][1] for w in walls_auto] + [w['end'][1] for w in walls_auto]
                                            min_x_auto, max_x_auto = min(all_x_auto), max(all_x_auto)
                                            min_y_auto, max_y_auto = min(all_y_auto), max(all_y_auto)
                                            
                                            scale_auto = int(st.session_state.viz_scale)
                                            margin_auto = 50
                                            img_height_auto = Image.open(io.BytesIO(st.session_state.viz_bytes)).height
                                            
                                            p1_auto, p2_auto = st.session_state.rect_coords
                                            x1_auto, y1_auto = min(p1_auto[0], p2_auto[0]), min(p1_auto[1], p2_auto[1])
                                            x2_auto, y2_auto = max(p1_auto[0], p2_auto[0]), max(p1_auto[1], p2_auto[1])
                                            
                                            rect_auto = {
                                                'left': x1_auto,
                                                'top': y1_auto,
                                                'width': x2_auto - x1_auto,
                                                'height': y2_auto - y1_auto
                                            }
                                            
                                            # 窓追加モードと線を結合モードは端点/交差ベースで検出（端点だけ囲む操作に対応）
                                            if edit_mode in ("窓を追加", "線を結合"):
                                                walls_in_rect_auto = _filter_walls_by_endpoints_in_rect(
                                                    walls_auto, rect_auto, scale_auto, margin_auto,
                                                    img_height_auto, min_x_auto, min_y_auto, max_x_auto, max_y_auto,
                                                    tolerance=0, debug=False
                                                )
                                            else:
                                                # 通常の線結合などは従来通り厳密判定
                                                walls_in_rect_auto = _filter_walls_strictly_in_rect(
                                                    walls_auto, rect_auto, scale_auto, margin_auto,
                                                    img_height_auto, min_x_auto, min_y_auto, max_x_auto, max_y_auto
                                                )

                                            # 2本の壁が検出された場合のみ自動追加（窓追加/線結合）
                                            if edit_mode in ("窓を追加", "線を結合"):
                                                if len(walls_in_rect_auto) == 2:
                                                    st.session_state.rect_coords_list.append((p1_auto, p2_auto))
                                                    st.session_state.rect_coords = []
                                                    st.session_state.last_click = None
                                                    try:
                                                        append_debug(f"Auto-added selection (2 walls detected): ids={[w.get('id') for w in walls_in_rect_auto]}")
                                                    except Exception:
                                                        pass
                                                    st.rerun()
                                                else:
                                                    # 端点が多く2本に絞れない場合、角度フィルタで2本に絞れれば自動追加する
                                                    try:
                                                        if len(walls_in_rect_auto) >= 2:
                                                            angle_threshold_preview = 30.0
                                                            angles = [math.radians(_wall_angle_deg(w)) for w in walls_in_rect_auto]
                                                            sx = sum(math.cos(a) for a in angles) if angles else 0
                                                            sy = sum(math.sin(a) for a in angles) if angles else 0
                                                            if sx == 0 and sy == 0:
                                                                avg_angle = 0.0
                                                            else:
                                                                avg_angle = math.degrees(math.atan2(sy, sx))

                                                            kept_preview = [w for w in walls_in_rect_auto if _angle_diff_deg(_wall_angle_deg(w), avg_angle) < angle_threshold_preview]
                                                            if len(kept_preview) == 2:
                                                                # 条件を満たすので自動で選択を追加
                                                                st.session_state.rect_coords_list.append((p1_auto, p2_auto))
                                                                st.session_state.rect_coords = []
                                                                st.session_state.last_click = None
                                                                try:
                                                                    append_debug(f"Auto-added selection (angle-filtered): kept_ids={[w.get('id') for w in kept_preview]}, avg_angle={avg_angle}")
                                                                except Exception:
                                                                    pass
                                                                st.rerun()
                                                    except Exception:
                                                        pass
                                            else:
                                                # オブジェクト配置モードでは四角形をそのまま追加（壁検出は不要）
                                                st.session_state.rect_coords_list.append((p1_auto, p2_auto))
                                                st.session_state.rect_coords = []
                                                st.session_state.last_click = None
                                                try:
                                                    append_debug(f"Auto-added object-placement selection: rect=({p1_auto},{p2_auto})")
                                                except Exception:
                                                    pass
                                                st.rerun()
                                        except Exception:
                                            pass
                                    
                                    st.rerun()  # 画像を再描画して選択点を表示
                
                    # 選択完了時のUI
                    if edit_mode == "線を削除" and len(st.session_state.rect_coords) == 2:
                        # 削除モード：2点選択完了（四角形）
                        p1, p2 = st.session_state.rect_coords
                        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                        st.success(f"✅ 2点選択完了: ({x1}, {y1}) - ({x2}, {y2})")
                    
                        with col_add:
                            if st.button("➕ この選択を追加", type="primary"):
                                # 現在の2点をリストに追加
                                st.session_state.rect_coords_list.append((p1, p2))
                                # 現在の選択をクリア
                                st.session_state.rect_coords = []
                                st.session_state.last_click = None
                                st.rerun()
                    elif edit_mode != "線を削除" and edit_mode != "線を結合" and len(st.session_state.rect_coords) == 2:
                        # 追加モード（線を結合以外）：2点選択完了
                        p1, p2 = st.session_state.rect_coords
                        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                        st.success(f"✅ 2点選択完了: ({x1}, {y1}) - ({x2}, {y2})")
                    
                        with col_add:
                            if st.button("➕ この選択を追加", type="primary"):
                                # 現在の2点をリストに追加
                                st.session_state.rect_coords_list.append((p1, p2))
                                # 現在の選択をクリア
                                st.session_state.rect_coords = []
                                st.session_state.last_click = None
                                st.rerun()
                    
                    # オブジェクト配置モード：家具のオプション選択を画像の下に表示
                    if edit_mode == "オブジェクトを配置" and len(st.session_state.rect_coords_list) > 0:
                        st.markdown("---")
                        st.markdown("### 🪑 家具のオプションを選択")
                        
                        col_height, col_color = st.columns(2)
                        
                        with col_height:
                            height_option = st.selectbox(
                                "高さ",
                                list(FURNITURE_HEIGHT_OPTIONS.keys()),
                                help="家具の高さを選択してください",
                                key="furniture_height_option"
                            )
                        
                        with col_color:
                            color_option = st.selectbox(
                                "配色",
                                list(FURNITURE_COLOR_OPTIONS.keys()),
                                help="家具の色を選択してください",
                                key="furniture_color_option"
                            )
                        
                        # 選択された高さを取得（天井合わせの場合は壁の高さ）
                        if height_option == "天井合わせ":
                            json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            walls = json_data['walls']
                            heights = [w.get('height', 2.4) for w in walls if 'height' in w]
                            selected_height = max(heights) if heights else 2.4
                            height_display = f"天井合わせ（{selected_height*100:.0f}cm）"
                        else:
                            selected_height = FURNITURE_HEIGHT_OPTIONS[height_option]
                            height_display = height_option
                        
                        # セッションステートに保存
                        st.session_state.furniture_params = {
                            'height_option': height_option,
                            'color_option': color_option,
                            'selected_height': selected_height
                        }
                        
                        # 選択された家具の情報を表示
                        #st.info(f"**{color_option}の家具**\n\n高さ: {height_display}")
                        
                        # 配置範囲のサイズを予測表示
                        if len(st.session_state.rect_coords_list) > 0:
                            rect = st.session_state.rect_coords_list[0]
                            p1, p2 = rect
                            json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                            x_start, y_start, width, depth = _snap_to_grid(
                                (p1[0], p1[1], p2[0], p2[1]), 
                                json_data, 
                                st.session_state.viz_scale
                            )
                            #st.success(f"📐 配置サイズ: 幅{width*100:.0f}cm × 奥行き{depth*100:.0f}cm × 高さ{selected_height*100:.0f}cm")
                        
                        if st.button("🪑 オブジェクト配置実行", type="primary", key="furniture_exec"):
                            st.session_state.execute_furniture_placement = True
                            st.rerun()
                    
                    # 確定済み選択の表示
                    # NOTE: ユーザー要望により、線を結合／線を削除／線を追加モードでは追加済みの選択範囲表示を抑制する
                    if len(st.session_state.rect_coords_list) > 0 and edit_mode not in ("線を結合", "線を削除", "線を追加", "オブジェクトを配置"):
                        if edit_mode == "線を削除":
                            st.markdown("### 📋 追加済みの削除対象")
                            for idx, (p1, p2) in enumerate(st.session_state.rect_coords_list):
                                color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][idx % 6]
                                st.write(f"#{idx+1}（{color_name}）: ({p1[0]}, {p1[1]})")
                        elif edit_mode == "窓を追加":
                            # 窓追加モードの場合、各選択範囲の壁検出状況を表示
                            try:
                                json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                walls = json_data['walls']
                                all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                                all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                                min_x, max_x = min(all_x), max(all_x)
                                min_y, max_y = min(all_y), max(all_y)
                                scale = int(viz_scale)
                                margin = 50
                                img_width = int((max_x - min_x) * scale) + 2 * margin
                                img_height = int((max_y - min_y) * scale) + 2 * margin
                                
                                # 全ての選択が成功しているかチェック
                                all_successful = True
                                for p1, p2 in st.session_state.rect_coords_list:
                                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                                    rect = {'left': x1, 'top': y1, 'width': x2 - x1, 'height': y2 - y1}
                                    # 窓追加モードおよび線を結合モードでは端点/交差ベースで検出（端点だけ囲む操作に対応）
                                    if edit_mode in ("窓を追加", "線を結合"):
                                        walls_in_rect = _filter_walls_by_endpoints_in_rect(
                                            walls, rect, scale, margin, img_height, min_x, min_y, max_x, max_y,
                                            tolerance=0, debug=False
                                        )
                                    else:
                                        walls_in_rect = _filter_walls_strictly_in_rect(
                                            walls, rect, scale, margin, img_height, min_x, min_y, max_x, max_y
                                        )
                                    if len(walls_in_rect) != 2:
                                        all_successful = False
                                        break
                                
                                # 失敗がある場合のみ見出しを表示
                                #if not all_successful:
                                #    st.markdown("### 📋 追加済みの選択範囲（窓）")
                                
                                #for idx, (p1, p2) in enumerate(st.session_state.rect_coords_list):
                                #    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                                #    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                                #    color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][idx % 6]
                                    
                                #    rect = {'left': x1, 'top': y1, 'width': x2 - x1, 'height': y2 - y1}
                                    # 窓追加モードおよび線を結合モードでは端点/交差ベースで検出（以前は厳密フィルタが使われていたため表示が不整合になっていた）
                                #    if edit_mode in ("窓を追加", "線を結合"):
                                #        walls_in_rect = _filter_walls_by_endpoints_in_rect(
                                #            walls, rect, scale, margin, img_height, min_x, min_y, max_x, max_y,
                                #            tolerance=0, debug=False
                                #        )
                                #    else:
                                #        walls_in_rect = _filter_walls_strictly_in_rect(
                                #            walls, rect, scale, margin, img_height, min_x, min_y, max_x, max_y
                                #        )
                                    
                                    # ハイライト表示（失敗の場合のみ）
                                #    if len(walls_in_rect) != 2:
                                #        if len(walls_in_rect) == 0:
                                #            st.error(f"#{idx+1}（{color_name}）: ({x1}, {y1}) - ({x2}, {y2})\n\n❌ 壁なし → 範囲を広げてください")
                                #        elif len(walls_in_rect) == 1:
                                #            st.warning(f"#{idx+1}（{color_name}）: ({x1}, {y1}) - ({x2}, {y2})\n\n⚠️ 1本のみ → 窓の両側の壁が入るように範囲を広げてください")
                                #        else:
                                #            st.warning(f"#{idx+1}（{color_name}）: ({x1}, {y1}) - ({x2}, {y2})\n\n⚠️ {len(walls_in_rect)}本（多すぎ） → 範囲を狭めてください")
                            except Exception as e:
                                st.error(f"壁検出エラー: {e}")
                            
                            # 窓追加モード: 各窓のパラメータ入力フォームを表示
                            if len(st.session_state.rect_coords_list) > 0:
                                st.markdown("---")
                                st.markdown("### 🪟 窓のサイズを入力")
                                st.info("💡 窓追加のパラメータを画面上部で入力してください（型番選択→窓高さ等）。")
                                
                                # 窓パラメータのデフォルト値を初期化
                                if 'window_params_list' not in st.session_state:
                                    st.session_state.window_params_list = []
                                
                                # リストのサイズを調整
                                while len(st.session_state.window_params_list) < len(st.session_state.rect_coords_list):
                                    st.session_state.window_params_list.append({
                                        'model': 'J4415/JF4415',
                                        'height_mm': 467,
                                        'base_mm': 1677
                                    })
                                
                                # 各窓の入力フォーム
                                for idx in range(len(st.session_state.rect_coords_list)):
                                    with st.expander(f"🪟 窓 #{idx+1} のパラメータ", expanded=True):
                                        col1, col2, col3 = st.columns([2, 1, 1])
                                        
                                        # 型番選択
                                        with col1:
                                            catalog_keys = sorted(list(WINDOW_CATALOG.keys()))
                                            catalog_keys.append("カスタム（手入力）")
                                            current_model = st.session_state.window_params_list[idx].get('model', 'J4415/JF4415')
                                            if current_model not in catalog_keys and current_model is not None:
                                                current_model = 'カスタム（手入力）'
                                            
                                            selected_model = st.selectbox(
                                                "型番（窓カタログ）",
                                                catalog_keys,
                                                index=catalog_keys.index(current_model) if current_model in catalog_keys else 0,
                                                key=f"window_model_{idx}"
                                            )
                                            
                                            # 型番が変更された場合、カタログ値で更新してrerun
                                            if selected_model != current_model:
                                                st.session_state.window_params_list[idx]['model'] = selected_model
                                                if selected_model != "カスタム（手入力）" and selected_model in WINDOW_CATALOG:
                                                    catalog_entry = WINDOW_CATALOG[selected_model]
                                                    if isinstance(catalog_entry, dict):
                                                        st.session_state.window_params_list[idx]['height_mm'] = int(catalog_entry.get('height', 1200))
                                                        st.session_state.window_params_list[idx]['base_mm'] = int(catalog_entry.get('base', 900))
                                                    else:
                                                        st.session_state.window_params_list[idx]['height_mm'] = int(catalog_entry)
                                                        st.session_state.window_params_list[idx]['base_mm'] = 900
                                                st.rerun()
                                            
                                            st.session_state.window_params_list[idx]['model'] = selected_model
                                        
                                        # 窓高さ
                                        with col2:
                                            # セッション状態から現在の値を取得
                                            current_h_mm = st.session_state.window_params_list[idx].get('height_mm', 1200)
                                            
                                            # 編集可能なnumber_input（型番も含めたkeyで一意性を確保）
                                            height_mm = st.number_input(
                                                "窓長さ(高さ) (mm)",
                                                min_value=50,
                                                max_value=3000,
                                                value=current_h_mm,
                                                step=1,
                                                key=f"window_height_{idx}_{selected_model}"
                                            )
                                            st.session_state.window_params_list[idx]['height_mm'] = height_mm
                                        
                                        # 床から窓下端
                                        with col3:
                                            # セッション状態から現在の値を取得
                                            current_base_mm = st.session_state.window_params_list[idx].get('base_mm', 900)
                                            
                                            # 編集可能なnumber_input（型番も含めたkeyで一意性を確保）
                                            base_mm = st.number_input(
                                                "床から窓下端 (mm)",
                                                min_value=0,
                                                max_value=5000,
                                                value=current_base_mm,
                                                step=1,
                                                key=f"window_base_{idx}_{selected_model}"
                                            )
                                            st.session_state.window_params_list[idx]['base_mm'] = base_mm
                                        
                                        # デバッグ情報を表示
                                        st.caption(f"💡 現在の設定: 高さ={st.session_state.window_params_list[idx]['height_mm']}mm, "
                                                  f"床から={st.session_state.window_params_list[idx]['base_mm']}mm, "
                                                  f"合計={st.session_state.window_params_list[idx]['height_mm'] + st.session_state.window_params_list[idx]['base_mm']}mm "
                                                  f"({(st.session_state.window_params_list[idx]['height_mm'] + st.session_state.window_params_list[idx]['base_mm'])/1000:.3f}m)")
                                
                                if st.button("🪟 窓追加実行", type="primary", key="window_batch_exec"):
                                    st.session_state.execute_window_batch = True
                                    st.rerun()
                        else:
                            st.markdown("### 📋 追加済みの選択範囲")
                            for idx, (p1, p2) in enumerate(st.session_state.rect_coords_list):
                                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                                color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][idx % 6]
                                st.write(f"#{idx+1}（{color_name}）: ({x1}, {y1}) - ({x2}, {y2})")
                
                    with col_exec:
                        # モード別のボタン表示と処理
                        # 永続デバッグログ表示（rerun しても残る）
                        # 永続デバッグログのUI表示は不要になったため削除（ログはセッションに保持）
                        if edit_mode == "線を結合":
                            button_label = "🔗 結合実行"
                        elif edit_mode == "窓を追加":
                            button_label = "🪟 窓追加実行"
                        elif edit_mode == "線を追加":
                            button_label = "➕ 線追加実行"
                        elif edit_mode == "オブジェクトを配置":
                            button_label = "🪑 オブジェクト配置実行"
                        elif edit_mode == "床を追加":
                            button_label = "🟫 床追加実行"
                        else:  # 線を削除
                            button_label = "🗑️ 削除実行"
                    
                        if edit_mode == "窓を追加":
                            # 窓追加モード：右側に重複して表示していた入力は削除
                            # 四角形が選択されている場合でも、画面上部のフォームで入力してください
                            if len(st.session_state.rect_coords_list) > 0:
                                # セッションから現在の窓高さを決定（mm->m 変換を優先）
                                if st.session_state.get('window_execution_params'):
                                    cur_wh_m = st.session_state['window_execution_params'].get('window_height', 1.2)
                                elif st.session_state.get('window_height_input_mm'):
                                    cur_wh_m = float(st.session_state['window_height_input_mm']) / 1000.0
                                elif st.session_state.get('window_height_display_mm'):
                                    cur_wh_m = float(st.session_state['window_height_display_mm']) / 1000.0
                                else:
                                    cur_wh_m = 1.2
                                
                                # 実行は画面上部のフォームで行うため、ここでは入力値の確認のみ表示
                                params_preview = st.session_state.get('window_execution_params', None)
                                if params_preview is not None:
                                    wh = params_preview.get('window_height', cur_wh_m)
                                    bh = params_preview.get('base_height', 0.9)
                                    bh_mm = params_preview.get('base_height_mm', int(bh * 1000))
                                    rh = params_preview.get('room_height', 2.4)
                                    ceiling_height = rh - (bh + wh)
                                    st.info(f"📐 床側の壁: {bh:.2f}m ({bh_mm}mm)、天井側の壁: {ceiling_height:.2f}m")
                                    if ceiling_height < 0:
                                        st.error("⚠️ 窓のサイズが部屋の高さを超えています")

                            # 追加: 現在の2点選択がある場合、実行前にデバッグ情報を表示
                            if len(st.session_state.get('rect_coords', [])) == 2:
                                # プレビュー時の詳細デバッグ表示は不要になったため非表示にします
                                try:
                                    p1, p2 = tuple(st.session_state['rect_coords'])
                                    if st.session_state.get('json_bytes'):
                                        json_data_preview = json.loads(st.session_state.json_bytes.decode('utf-8'))
                                        walls_preview = json_data_preview.get('walls', [])
                                        all_x = [w['start'][0] for w in walls_preview] + [w['end'][0] for w in walls_preview]
                                        all_y = [w['start'][1] for w in walls_preview] + [w['end'][1] for w in walls_preview]
                                        min_x, max_x = min(all_x), max(all_x)
                                        min_y, max_y = min(all_y), max(all_y)
                                        scale_preview = int(st.session_state.get('viz_scale', 100))
                                        margin_preview = 50
                                        img_height_preview = int((max_y - min_y) * scale_preview) + 2 * margin_preview
                                        rect_preview = {
                                            'left': min(p1[0], p2[0]),
                                            'top': min(p1[1], p2[1]),
                                            'width': abs(p2[0] - p1[0]),
                                            'height': abs(p2[1] - p1[1])
                                        }
                                        # 内部検証は行うが表示は行わない（必要ならログに出す）
                                        walls_hit, debug_info_preview = _filter_walls_by_endpoints_in_rect(
                                            walls_preview, rect_preview, scale_preview, margin_preview, img_height_preview,
                                            min_x, min_y, max_x, max_y, tolerance=0, debug=True
                                        )
                                        try:
                                            append_debug(f"Window-add preview: detected_ids={[w.get('id') for w in walls_hit]}, total_checked={len(debug_info_preview)}")
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                        
                        # 窓追加モードで一括実行ボタンが押された場合の処理
                        if edit_mode == "窓を追加" and (st.session_state.get('execute_window_now') or st.session_state.get('execute_window_batch')):
                            # フラグをクリア
                            st.session_state.execute_window_now = False
                            st.session_state.execute_window_batch = False
                            
                            try:
                                try:
                                    append_debug(f"Window-add execute: target_rects_count={len(st.session_state.rect_coords_list)}, window_params_count={len(st.session_state.get('window_params_list', []))}")
                                except Exception:
                                    pass
                                
                                # 処理対象の四角形リストを作成
                                target_rects = list(st.session_state.rect_coords_list)
                                window_params_list = st.session_state.get('window_params_list', [])
                                
                                if len(target_rects) == 0:
                                    st.error("⚠️ 窓の範囲を選択してください")
                                    st.stop()
                                
                                if len(window_params_list) < len(target_rects):
                                    st.error("⚠️ すべての窓のパラメータを入力してください")
                                    st.stop()
                            
                                # JSONデータを読み込み
                                json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                walls = json_data['walls']
                            
                                # 可視化画像のパラメータを取得
                                all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                                all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                                min_x, max_x = min(all_x), max(all_x)
                                min_y, max_y = min(all_y), max(all_y)
                            
                                scale = int(viz_scale)
                                margin = 50
                                img_width = int((max_x - min_x) * scale) + 2 * margin
                                img_height = int((max_y - min_y) * scale) + 2 * margin
                            
                                # 編集前の画像を保存（比較用）
                                original_viz_bytes = st.session_state.viz_bytes
                            
                                # 元データを保護するためディープコピー
                                import copy
                                original_json_data = copy.deepcopy(json_data)
                                updated_json = json_data
                                
                                # 追加した壁のID（赤色表示用）
                                added_wall_ids = []
                                
                                # 天井高さ（部屋の高さ）を取得
                                heights = [w.get('height', 2.4) for w in walls if 'height' in w]
                                room_height = max(heights) if heights else 2.4
                                
                                # ===== 窓を追加モード =====
                                # デバッグが必要なため、詳細ログはデフォルトで展開表示する
                                with st.expander("🔍 窓追加処理の詳細ログ", expanded=True):
                                    total_added_count = 0
                                    window_details = []
                    
                                    for rect_idx, (p1, p2) in enumerate(target_rects):
                                        st.markdown(f"---\n\n**窓#{rect_idx+1}の処理:**")
                                        
                                        # この窓のパラメータを取得
                                        if rect_idx < len(window_params_list):
                                            window_param = window_params_list[rect_idx]
                                            window_height_mm = window_param.get('height_mm', 1200)
                                            base_height_mm = window_param.get('base_mm', 900)
                                            window_model = window_param.get('model', None)
                                            
                                            window_height = float(window_height_mm) / 1000.0
                                            base_height = float(base_height_mm) / 1000.0
                                            
                                            st.info(f"📝 型番: {window_model if window_model and window_model != 'カスタム（手入力）' else 'カスタム'}, "
                                                   f"高さ={window_height}m ({window_height_mm}mm), "
                                                   f"床から={base_height}m ({base_height_mm}mm)")
                                        else:
                                            st.error(f"⚠️ 窓#{rect_idx+1}のパラメータが見つかりません")
                                            continue
                                        
                                        rect = {
                                            'left': min(p1[0], p2[0]),
                                            'top': min(p1[1], p2[1]),
                                            'width': abs(p2[0] - p1[0]),
                                            'height': abs(p2[1] - p1[1])
                                        }
                                    
                                        # 四角形内に端点がある壁線を抽出（窓追加モード：端点2つだけを囲めばOK）
                                        # デバッグモードで詳細情報を取得
                                        walls_in_rect, debug_info = _filter_walls_by_endpoints_in_rect(
                                            updated_json['walls'], rect, scale, margin, img_height, min_x, min_y, max_x, max_y, 
                                            tolerance=0, debug=True
                                        )
                                        
                                        # デバッグ情報はUI表示を控え、セッションログへ出力
                                        try:
                                            append_debug(f"Window rect #{rect_idx+1}: coords=({p1},{p2}), walls_hit_count={len(walls_in_rect)}")
                                            append_debug(f"Window rect coords: p1={p1}, p2={p2}")
                                            for info in debug_info:
                                                append_debug(f"Window rect debug: {info}")
                                        except Exception:
                                            pass
                                        
                                        # 全ての壁のIDリストも表示（参考用）
                                        all_wall_ids = [w.get('id', '?') for w in updated_json['walls']]
                                        st.write(f"**全壁ID一覧:** {all_wall_ids[:20]}{'...' if len(all_wall_ids) > 20 else ''} (合計{len(all_wall_ids)}本)")
                                    
                                        st.write(f"**選択範囲内の壁:** {len(walls_in_rect)}本")
                                        if walls_in_rect:
                                            st.write(f"検出された壁ID: {[w['id'] for w in walls_in_rect]}")
                                        # フォールバック: 端点で複数検出された場合、角度フィルタで2本に絞れるならそれを使う
                                        if len(walls_in_rect) != 2 and len(walls_in_rect) >= 2:
                                            try:
                                                angle_threshold_preview = 30.0
                                                angles = [math.radians(_wall_angle_deg(w)) for w in walls_in_rect]
                                                sx = sum(math.cos(a) for a in angles) if angles else 0
                                                sy = sum(math.sin(a) for a in angles) if angles else 0
                                                if sx == 0 and sy == 0:
                                                    avg_angle = 0.0
                                                else:
                                                    avg_angle = math.degrees(math.atan2(sy, sx))
                                                kept = [w for w in walls_in_rect if _angle_diff_deg(_wall_angle_deg(w), avg_angle) < angle_threshold_preview]
                                                if len(kept) == 2:
                                                    try:
                                                        append_debug(f"Window execution: angle-filtered walls from {len(walls_in_rect)} to 2: ids={[w.get('id') for w in kept]}, avg_angle={avg_angle}")
                                                    except Exception:
                                                        pass
                                                    walls_in_rect = kept
                                                    # 表示も更新しておく
                                                    st.write(f"検出された壁ID(フィルタ後): {[w['id'] for w in walls_in_rect]}")
                                            except Exception:
                                                pass
                                    
                                        if len(walls_in_rect) == 2:
                                            # 2本の壁の間に床側と天井側の壁を追加
                                            st.success(f"✅ 2本の壁を検出、窓追加処理を実行します")
                                            st.write(f"**デバッグ:** window_height={window_height}m ({window_height_mm}mm), base_height={base_height}m ({base_height_mm}mm), room_height={room_height}m")
                                            st.write(f"**計算:** ceiling_height = {room_height} - ({base_height} + {window_height}) = {room_height - (base_height + window_height)}m")
                                            updated_json, added_walls = _add_window_walls(
                                                updated_json,
                                                walls_in_rect[0],
                                                walls_in_rect[1],
                                                window_height,
                                                base_height,
                                                room_height,
                                                window_model if window_model != 'カスタム（手入力）' else None,
                                                window_height_mm
                                            )
                                            total_added_count += len(added_walls)
                                            st.success(f"✅ {len(added_walls)}本の壁を追加しました（ID: {[w['id'] for w in added_walls]}）")
                                            
                                            # 追加した壁の詳細を表示
                                            for aw in added_walls:
                                                st.write(f"  追加壁ID#{aw['id']}: height={aw.get('height')}m ({aw.get('height')*1000:.0f}mm), "
                                                        f"base_height={aw.get('base_height')}m ({aw.get('base_height')*1000:.0f}mm)")
                                            
                                            # 追加した壁のIDを記録（赤色表示用）
                                            added_wall_ids.extend([w['id'] for w in added_walls])
                                        
                                            color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][rect_idx % 6]
                                            window_details.append({
                                                'rect_idx': rect_idx,
                                                'color_name': color_name,
                                                'wall_ids': [w['id'] for w in added_walls],
                                                'window_height': window_height,
                                                'window_height_mm': window_height_mm,
                                                'window_model': window_model if window_model != 'カスタム（手入力）' else None,
                                                'base_height': base_height,
                                                'base_height_mm': base_height_mm
                                            })
                                        elif len(walls_in_rect) < 2:
                                            try:
                                                append_debug(f"Window rect #{rect_idx+1} skipped during execution: found {len(walls_in_rect)} walls")
                                            except Exception:
                                                pass
                                            st.warning(f"⚠️ 四角形#{rect_idx+1}: 2本の壁が必要ですが、{len(walls_in_rect)}本しか見つかりません")
                                        else:
                                            st.warning(f"⚠️ 四角形#{rect_idx+1}: 2本の壁を選択してください（{len(walls_in_rect)}本選択されています）")
                    
                                    if total_added_count > 0:
                                        st.success(f"✅✅ 合計 {total_added_count} 本の壁を追加しました（窓{len(window_details)}箇所）")
                                    
                                        # 追加詳細を表示
                                        st.markdown("**窓追加結果:**")
                                        for detail in window_details:
                                            st.write(
                                                f"#{detail['rect_idx']+1}（{detail['color_name']}）: "
                                                f"壁({detail['wall_ids'][0]}, {detail['wall_ids'][1]}) を追加 - "
                                                f"窓高さ: {detail['window_height']}m ({detail.get('window_height_mm', int(detail['window_height']*1000))}mm), 床から: {detail['base_height']}m ({int(detail.get('base_height_mm', detail['base_height']*1000))}mm)"
                                            )
                                    else:
                                        st.warning("⚠️ 追加可能な窓が見つかりません")
                                
                                # 一時ファイルに保存
                                temp_json_path = Path(st.session_state.out_dir) / "walls_3d_edited.json"
                                with open(temp_json_path, 'w', encoding='utf-8') as f:
                                    json.dump(updated_json, f, indent=2, ensure_ascii=False)
                                
                                # 再可視化（元の変換と同じスケールを使用）
                                # 追加した壁を赤色で表示
                                temp_viz_path = Path(st.session_state.out_dir) / "visualization_edited.png"
                                visualize_3d_walls(str(temp_json_path), str(temp_viz_path), scale=int(viz_scale), highlight_wall_ids=added_wall_ids, wall_color=(0, 0, 0), bg_color=(255, 255, 255))
                            
                                # 3Dビューア生成
                                temp_viewer_path = Path(st.session_state.out_dir) / "viewer_3d_edited.html"
                                _generate_3d_viewer_html(temp_json_path, temp_viewer_path)
                            
                                # 窓追加モードは自動保存
                                # セッション状態を更新（JSON・画像・3Dビューア）
                                st.session_state.json_bytes = temp_json_path.read_bytes()
                                st.session_state.json_name = "walls_3d_edited.json"
                                st.session_state.viz_bytes = temp_viz_path.read_bytes()
                                
                                # 3DビューアHTMLも更新
                                st.session_state.viewer_html_bytes = temp_viewer_path.read_bytes()
                                st.session_state.viewer_html_name = temp_viewer_path.name
                                
                                # 編集状態を完全にクリア
                                st.session_state.rect_coords = []
                                st.session_state.rect_coords_list = []
                                st.session_state.last_click = None
                                st.session_state.reset_flag = False
                                st.session_state.merge_result = None
                                
                                # 窓追加パラメータもクリア
                                if 'window_execution_params' in st.session_state:
                                    del st.session_state.window_execution_params
                                
                                st.success("✅ 窓追加完了！自動保存しました。さらに編集を続けることができます。")
                                time.sleep(0.5)
                                st.rerun()
                        
                            except Exception as e:
                                st.error(f"エラーが発生しました: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                        
                        elif edit_mode != "窓を追加" and (len(st.session_state.rect_coords_list) > 0 or len(st.session_state.rect_coords) == 2):
                            # 結合・追加・削除モードの実行ボタン（窓追加モードは上記で別処理）
                            should_execute = False
                            if st.button(button_label, type="primary", key="btn_general_edit_exec"):
                                should_execute = True
                            
                            if should_execute:
                                try:
                                    # 処理対象の四角形リストを作成（確定済み選択 + 現在選択中の2点）
                                    target_rects = list(st.session_state.rect_coords_list)
                                    if len(st.session_state.rect_coords) == 2:
                                        target_rects.append(tuple(st.session_state.rect_coords))
                                
                                    # JSONデータを読み込み
                                    json_data = json.loads(st.session_state.json_bytes.decode("utf-8"))
                                    walls = json_data['walls']
                                
                                    # 可視化画像のパラメータを取得
                                    all_x = [w['start'][0] for w in walls] + [w['end'][0] for w in walls]
                                    all_y = [w['start'][1] for w in walls] + [w['end'][1] for w in walls]
                                    min_x, max_x = min(all_x), max(all_x)
                                    min_y, max_y = min(all_y), max(all_y)
                                
                                    scale = int(viz_scale)
                                    margin = 50
                                    img_width = int((max_x - min_x) * scale) + 2 * margin
                                    img_height = int((max_y - min_y) * scale) + 2 * margin
                                
                                    # 編集前の画像を保存（比較用）
                                    original_viz_bytes = st.session_state.viz_bytes
                                
                                    # 元データを保護するためディープコピー
                                    import copy
                                    original_json_data = copy.deepcopy(json_data)
                                    updated_json = json_data
                                    
                                    # 追加した壁のID（窓追加モードで使用）
                                    added_wall_ids = []
                                
                                    # 各モード用の変数を事前初期化
                                    total_merged_count = 0
                                    merge_details = []
                                    total_added_count = 0
                                    add_details = []
                                    total_deleted_count = 0
                                    delete_details = []
                                    total_floor_count = 0
                                    floor_details = []
                                    
                                    if edit_mode == "オブジェクトを配置":
                                        # ===== オブジェクトを配置モード =====
                                        # セッションステートから家具パラメータを取得
                                        furniture_params = st.session_state.get('furniture_params', {})
                                        height_option = furniture_params.get('height_option', '30cm')
                                        color_option = furniture_params.get('color_option', 'ダーク')
                                        
                                        # 高さオプションの取得
                                        if height_option == "天井合わせ":
                                            heights = [w.get('height', 2.4) for w in walls if 'height' in w]
                                            furniture_height = max(heights) if heights else 2.4
                                        else:
                                            furniture_height = FURNITURE_HEIGHT_OPTIONS.get(height_option, 0.3)
                                        
                                        # 各四角形をループして処理
                                        for rect_idx, (p1, p2) in enumerate(target_rects):
                                            # 四角形範囲をメートル座標に変換
                                            x_start, y_start, width, depth = _snap_to_grid(
                                                (p1[0], p1[1], p2[0], p2[1]),
                                                json_data,
                                                scale
                                            )
                                            
                                            # 家具を追加
                                            updated_json = _add_furniture_to_json(
                                                updated_json,
                                                furniture_height,
                                                color_option,
                                                x_start,
                                                y_start,
                                                width,
                                                depth
                                            )
                                        
                                        # 自動保存: オブジェクト配置結果を JSON/可視化/3Dビューアに反映
                                        try:
                                            temp_json_path = Path(st.session_state.out_dir) / "walls_3d_edited.json"
                                            with open(temp_json_path, 'w', encoding='utf-8') as f:
                                                json.dump(updated_json, f, ensure_ascii=False, indent=2)

                                            temp_viz_path = Path(st.session_state.out_dir) / "visualization_edited.png"
                                            visualize_3d_walls(str(temp_json_path), str(temp_viz_path), scale=int(viz_scale), highlight_wall_ids=added_wall_ids, wall_color=(0, 0, 0), bg_color=(255, 255, 255))

                                            temp_viewer_path = Path(st.session_state.out_dir) / "viewer_3d_edited.html"
                                            _generate_3d_viewer_html(temp_json_path, temp_viewer_path)

                                            # セッションに保存して UI 上でダウンロード可能にする
                                            st.session_state.json_bytes = temp_json_path.read_bytes()
                                            st.session_state.json_name = temp_json_path.name
                                            st.session_state.viz_bytes = temp_viz_path.read_bytes()
                                            st.session_state.viewer_html_bytes = temp_viewer_path.read_bytes()
                                            st.session_state.viewer_html_name = temp_viewer_path.name

                                            # 選択状態をクリア
                                            st.session_state.rect_coords = []
                                            st.session_state.rect_coords_list = []
                                            st.session_state.last_click = None
                                        except Exception as e:
                                            st.error(f"保存エラー: {e}")
                                        
                                    elif edit_mode == "線を結合":
                                        # ===== 線を結合モード =====
                                        # 各四角形をループして処理
                                        total_merged_count = 0
                                        merge_details = []
                                    
                                        for rect_idx, (p1, p2) in enumerate(target_rects):
                                            rect = {
                                                'left': min(p1[0], p2[0]),
                                                'top': min(p1[1], p2[1]),
                                                'width': abs(p2[0] - p1[0]),
                                                'height': abs(p2[1] - p1[1])
                                            }
                                            try:
                                                append_debug(f"Merge started: rect_idx={rect_idx+1}, rect={rect}")
                                            except Exception:
                                                pass
                                        
                                            # 選択範囲内の壁線を抽出
                                            # スケール校正版と同様に「完全に含まれる」判定をまず試す（端点ベース）
                                            walls_in_selection = _filter_walls_strictly_in_rect(
                                                updated_json['walls'], rect, scale, margin, img_height, min_x, min_y, max_x, max_y
                                            )
                                            # 端点ベースで何も見つからなければ、交差/近接ベースでフォールバック（これを無効にする場合は削除）
                                            if len(walls_in_selection) == 0:
                                                walls_in_selection = [
                                                    wall for wall in updated_json['walls']
                                                    if _wall_in_rect(wall, rect, scale, margin, img_height, min_x, min_y, max_x, max_y)
                                                ]
                                            try:
                                                append_debug(f"walls_in_selection ids: {[w.get('id') for w in walls_in_selection]} (count={len(walls_in_selection)})")
                                            except Exception:
                                                pass

                                            # プレビューでフィルタ済みIDがセッションに保存されている場合、
                                            # 四角形が一致すれば execution 側の検出集合をプレビューのフィルタ済み集合に合わせる。
                                            try:
                                                last_filtered = st.session_state.get('last_preview_filtered_ids')
                                                last_rect = st.session_state.get('last_preview_rect')
                                                if last_filtered and last_rect:
                                                    if (abs(last_rect.get('left',0) - rect.get('left',0)) < 1 and
                                                        abs(last_rect.get('top',0) - rect.get('top',0)) < 1 and
                                                        abs(last_rect.get('width',0) - rect.get('width',0)) < 1 and
                                                        abs(last_rect.get('height',0) - rect.get('height',0)) < 1):
                                                        # updated_json の walls から該当IDを抽出（四角形外でも preview が見ていたIDを優先）
                                                        id_set = set(last_filtered)
                                                        walls_in_selection = [w for w in updated_json['walls'] if w.get('id') in id_set]
                                                        try:
                                                            st.write(f"🔧 プレビューのフィルタ済みIDを優先して walls_in_selection を置換しました: {list(id_set)}")
                                                        except Exception:
                                                            pass
                                                        try:
                                                            append_debug(f"Applied preview filtered ids as walls_in_selection: {list(id_set)}")
                                                        except Exception:
                                                            pass
                                            except Exception:
                                                pass
                                        
                                            # まず、プレビューで選ばれたペアがあるかを確認し、四角形が一致すればそれを優先する
                                            walls_to_use = None
                                            try:
                                                last_pair = st.session_state.get('last_preview_pair')
                                                last_rect = st.session_state.get('last_preview_rect')
                                                if last_pair and last_rect:
                                                    # rect と同じなら preview のペアを利用
                                                    if (abs(last_rect.get('left',0) - rect.get('left',0)) < 1 and
                                                        abs(last_rect.get('top',0) - rect.get('top',0)) < 1 and
                                                        abs(last_rect.get('width',0) - rect.get('width',0)) < 1 and
                                                        abs(last_rect.get('height',0) - rect.get('height',0)) < 1):
                                                        # updated_json の walls から id を探して walls_to_use を構築
                                                        id_set = set(last_pair)
                                                        walls_to_use = [w for w in walls_in_selection if w.get('id') in id_set]
                                                        if len(walls_to_use) != 2:
                                                            walls_to_use = None
                                            except Exception:
                                                walls_to_use = None

                                            # プレビュー優先が使えない場合は従来通り角度フィルタを試す
                                            if walls_to_use is None:
                                                filtered_by_angle = None
                                                try:
                                                    angles = [math.radians(_wall_angle_deg(w)) for w in walls_in_selection]
                                                    sx = sum(math.cos(a) for a in angles)
                                                    sy = sum(math.sin(a) for a in angles)
                                                    if sx == 0 and sy == 0:
                                                        avg_angle = 0.0
                                                    else:
                                                        avg_angle = math.degrees(math.atan2(sy, sx))

                                                    angle_threshold = 30.0
                                                    kept = [w for w in walls_in_selection if _angle_diff_deg(_wall_angle_deg(w), avg_angle) < angle_threshold]
                                                    if len(kept) >= 2:
                                                        filtered_by_angle = kept
                                                except Exception:
                                                    filtered_by_angle = None

                                                if filtered_by_angle is not None:
                                                    walls_to_use = filtered_by_angle
                                                else:
                                                    walls_to_use = walls_in_selection

                                            try:
                                                append_debug(f"walls_to_use ids: {[w.get('id') for w in walls_to_use]} (count={len(walls_to_use)})")
                                            except Exception:
                                                pass

                                            # デバッグログにも追加
                                            try:
                                                wall_info = f"選択範囲内の壁: {len(walls_to_use)}本"
                                                st.write(f"**{wall_info}**")
                                                append_debug(wall_info)
                                                if walls_to_use:
                                                    wall_ids_in_selection = [w['id'] for w in walls_to_use]
                                                    wall_display = ", ".join([f"壁({wid})" for wid in wall_ids_in_selection])
                                                    wall_list_info = f"壁: {wall_display}"
                                                    st.write(wall_list_info)
                                                    append_debug(wall_list_info)
                                            except Exception:
                                                pass
                                        
                                            if len(walls_to_use) >= 2:
                                                # 複数線が選択されている場合、方向を判定して最も離れた2本のペアのみを結合
                                                # 四角形の幅と高さから方向を判定
                                                rect_width = abs(p2[0] - p1[0])
                                                rect_height = abs(p2[1] - p1[1])
                                            
                                                if rect_width > rect_height:
                                                    # X方向：x座標で最も離れた2本を選択
                                                    walls_by_x = sorted(walls_to_use, 
                                                                        key=lambda w: min(w['start'][0], w['end'][0]))
                                                    leftmost_wall = walls_by_x[0]
                                                    rightmost_wall = walls_by_x[-1]
                                                
                                                    # 2本だけを結合候補として抽出
                                                    selected_walls = [leftmost_wall, rightmost_wall]
                                                    direction = "X方向"
                                                else:
                                                    # Y方向：y座標で最も離れた2本を選択
                                                    walls_by_y = sorted(walls_to_use,
                                                                    key=lambda w: min(w['start'][1], w['end'][1]))
                                                    bottom_wall = walls_by_y[0]
                                                    top_wall = walls_by_y[-1]
                                                
                                                    # 2本だけを結合候補として抽出
                                                    selected_walls = [bottom_wall, top_wall]
                                                    direction = "Y方向"
                                            
                                                # デバッグログにも追加
                                                try:
                                                    direction_info = f"方向判定: {direction} (幅: {rect_width}px, 高さ: {rect_height}px)"
                                                    merge_target_info = f"結合対象: 壁({selected_walls[0]['id']}) ← → 壁({selected_walls[1]['id']})"
                                                    st.write(f"**{direction_info}**")
                                                    st.write(f"**{merge_target_info}**")
                                                    append_debug(direction_info)
                                                    append_debug(merge_target_info)
                                                except Exception:
                                                    pass
                                            
                                                # 結合候補を探す（選択された2本だけ）
                                                # 結合側の閾値はプレビューの角度フィルタと合わせて30度に緩和
                                                merge_angle_threshold = 30
                                                # 追加デバッグ: 選択集合内の全ペアについて最短端点距離と角度差を表示
                                                try:
                                                    pair_debug = []
                                                    for i, wa in enumerate(walls_to_use):
                                                        for j, wb in enumerate(walls_to_use):
                                                            if i >= j:
                                                                continue
                                                            endpoints_a = [wa['start'], wa['end']]
                                                            endpoints_b = [wb['start'], wb['end']]
                                                            min_d = None
                                                            for pa in endpoints_a:
                                                                for pb in endpoints_b:
                                                                    d = _calc_distance(pa, pb)
                                                                    if min_d is None or d < min_d:
                                                                        min_d = d
                                                            ang = _calc_angle_diff(wa, wb)
                                                            pair_debug.append({'wall1': wa.get('id'), 'wall2': wb.get('id'), 'min_endpoint_dist_m': round(min_d,4) if min_d is not None else None, 'angle_diff_deg': round(ang,2)})
                                                    pair_debug_str = str(pair_debug)
                                                    st.write('**デバッグ (全ペア距離/角度):**', pair_debug)
                                                    try:
                                                        append_debug(f"全ペア距離/角度: {pair_debug_str}")
                                                    except Exception:
                                                        pass
                                                except Exception:
                                                    pass

                                                # チェーン検出を含めるため、選択された2本のみではなく
                                                # フィルタ済みの `walls_to_use` 全体を渡す。
                                                candidates = _find_mergeable_walls(
                                                    walls_to_use,
                                                    distance_threshold=distance_threshold,
                                                    angle_threshold=merge_angle_threshold
                                                )

                                                # デバッグ情報: 選択した2本の角度差と最短端点距離を表示
                                                try:
                                                    angle_diff_sel = _calc_angle_diff(selected_walls[0], selected_walls[1])
                                                    # 最短端点距離を計算
                                                    endpoints1 = [selected_walls[0]['start'], selected_walls[0]['end']]
                                                    endpoints2 = [selected_walls[1]['start'], selected_walls[1]['end']]
                                                    min_dist = min(_calc_distance(p1, p2) for p1 in endpoints1 for p2 in endpoints2)
                                                    selected_wall_info = f"選択壁の角度差: {angle_diff_sel:.2f}度, 最短端点距離: {min_dist:.3f} m"
                                                    st.write(f"**{selected_wall_info}**")
                                                    try:
                                                        append_debug(selected_wall_info)
                                                    except Exception:
                                                        pass
                                                except Exception:
                                                    pass

                                                # 候補の詳細を常に表示（空でも明示）
                                                try:
                                                    cand_list = []
                                                    for c in candidates:
                                                        if c.get('is_chain'):
                                                            cand_list.append({'type': 'chain', 'chain_length': c.get('chain_length'), 'distance': c.get('distance'), 'angle_diff': c.get('angle_diff'), 'confidence': c.get('confidence')})
                                                        else:
                                                            cand_list.append({'type': 'pair', 'wall1': c.get('wall1', {}).get('id'), 'wall2': c.get('wall2', {}).get('id'), 'distance': c.get('distance'), 'angle_diff': c.get('angle_diff'), 'confidence': c.get('confidence')})
                                                    cand_list_str = str(cand_list)
                                                    st.write("**デバッグ (候補一覧):**", cand_list)
                                                    try:
                                                        append_debug(f"候補一覧: {cand_list_str}")
                                                    except Exception:
                                                        pass
                                                except Exception:
                                                    pass

                                                # フォールバック: 候補が見つからなければ閾値を緩めて再探索
                                                if not candidates:
                                                    try:
                                                        fallback_dist = max(distance_threshold * 2, 0.5)
                                                        fallback_angle = max(merge_angle_threshold * 2, 45)
                                                        st.warning(f"候補が見つかりません。フォールバック閾値で再探索します (距離: {fallback_dist}m, 角度: {fallback_angle}°)")
                                                        candidates_fb = _find_mergeable_walls(
                                                            selected_walls,
                                                            distance_threshold=fallback_dist,
                                                            angle_threshold=fallback_angle
                                                        )
                                                        cand_fb_list = []
                                                        for c in candidates_fb:
                                                            if c.get('is_chain'):
                                                                cand_fb_list.append({'type': 'chain', 'chain_length': c.get('chain_length'), 'distance': c.get('distance'), 'angle_diff': c.get('angle_diff'), 'confidence': c.get('confidence')})
                                                            else:
                                                                cand_fb_list.append({'type': 'pair', 'wall1': c.get('wall1', {}).get('id'), 'wall2': c.get('wall2', {}).get('id'), 'distance': c.get('distance'), 'angle_diff': c.get('angle_diff'), 'confidence': c.get('confidence')})
                                                        st.write("**デバッグ (フォールバック候補一覧):**", cand_fb_list)
                                                        if candidates_fb:
                                                            candidates = candidates_fb
                                                            st.info("フォールバックで候補が見つかりました。結合を実行します。")
                                                            try:
                                                                append_debug(f"Fallback candidates: {cand_fb_list}")
                                                            except Exception:
                                                                pass
                                                        else:
                                                            st.warning("フォールバックでも候補が見つかりませんでした。四角形選択や閾値を確認してください。")
                                                    except Exception:
                                                        pass

                                                # 追加: 2点選択時に候補が見つからない場合、周辺の壁を含めたチェーン検索を試みる
                                                if not candidates:
                                                    try:
                                                        # selected_walls が存在し、2本選択されている場合にのみ実行
                                                        if 'selected_walls' in locals() and len(selected_walls) == 2:
                                                            # 参照角度と閾値
                                                            ref_angle = _wall_angle_deg(selected_walls[0])
                                                            angle_tol = merge_angle_threshold
                                                            # 周辺壁を収集（角度が近く、端点距離が近いもの）
                                                            neighborhood = []
                                                            for w in updated_json.get('walls', []):
                                                                try:
                                                                    if _angle_diff_deg(_wall_angle_deg(w), ref_angle) < angle_tol:
                                                                        endpoints_sel = [selected_walls[0]['start'], selected_walls[0]['end'], selected_walls[1]['start'], selected_walls[1]['end']]
                                                                        endpoints_w = [w['start'], w['end']]
                                                                        min_d = min(_calc_distance(p1, p2) for p1 in endpoints_sel for p2 in endpoints_w)
                                                                        # 距離閾値は少し広めに設定（既定閾値の2倍または0.5m）
                                                                        if min_d <= max(distance_threshold * 2, 0.5):
                                                                            neighborhood.append(w)
                                                                except Exception:
                                                                    continue

                                                            if len(neighborhood) >= 2:
                                                                candidates_ext = _find_mergeable_walls(
                                                                    neighborhood,
                                                                    distance_threshold=distance_threshold,
                                                                    angle_threshold=merge_angle_threshold
                                                                )
                                                                if candidates_ext:
                                                                    candidates = candidates_ext
                                                                    st.info('周辺壁を含めた拡張チェーン検索で候補を検出しました。')
                                                                    try:
                                                                        append_debug(f"Neighborhood-extended candidates found: count={len(candidates_ext)}")
                                                                    except Exception:
                                                                        pass
                                                    except Exception:
                                                        pass
                                                # 追加フォールバック: 2点選択で角度が近いが距離が大きい場合、小さめの閾値で自動強制候補を作成する
                                                if not candidates:
                                                    try:
                                                        if 'selected_walls' in locals() and len(selected_walls) == 2:
                                                            # 最短端点距離と角度差を計算
                                                            endpoints1 = [selected_walls[0]['start'], selected_walls[0]['end']]
                                                            endpoints2 = [selected_walls[1]['start'], selected_walls[1]['end']]
                                                            min_dist_tmp = min(_calc_distance(p1, p2) for p1 in endpoints1 for p2 in endpoints2)
                                                            angle_diff_tmp = _calc_angle_diff(selected_walls[0], selected_walls[1])
                                                            # 許容距離：ほぼ無制限にする（既定閾値の100倍）
                                                            extended_limit = distance_threshold * 100
                                                            if angle_diff_tmp <= merge_angle_threshold and min_dist_tmp <= extended_limit:
                                                                # 自動強制候補を作成
                                                                w1 = selected_walls[0]
                                                                w2 = selected_walls[1]
                                                                # 最短端点組合せから接続タイプを決定
                                                                min_pair = None
                                                                md = None
                                                                for p1 in endpoints1:
                                                                    for p2 in endpoints2:
                                                                        d = _calc_distance(p1, p2)
                                                                        if md is None or d < md:
                                                                            md = d
                                                                            min_pair = (p1, p2)
                                                                p1p, p2p = min_pair
                                                                conn = 'end-start'
                                                                if p1p == w1.get('end') and p2p == w2.get('start'):
                                                                    conn = 'end-start'
                                                                elif p1p == w1.get('end') and p2p == w2.get('end'):
                                                                    conn = 'end-end'
                                                                elif p1p == w1.get('start') and p2p == w2.get('start'):
                                                                    conn = 'start-start'
                                                                elif p1p == w1.get('start') and p2p == w2.get('end'):
                                                                    conn = 'start-end'
                                                                # new_start/new_end を接続タイプに合わせて設定
                                                                if conn == 'end-start':
                                                                    new_start = w1.get('start')
                                                                    new_end = w2.get('end')
                                                                elif conn == 'end-end':
                                                                    new_start = w1.get('start')
                                                                    new_end = w2.get('start')
                                                                elif conn == 'start-start':
                                                                    new_start = w1.get('end')
                                                                    new_end = w2.get('end')
                                                                elif conn == 'start-end':
                                                                    new_start = w1.get('end')
                                                                    new_end = w2.get('start')
                                                                else:
                                                                    new_start = None
                                                                    new_end = None
                                                                forced_candidate_auto = {
                                                                    'wall1': w1,
                                                                    'wall2': w2,
                                                                    'is_chain': False,
                                                                    'distance': md,
                                                                    'angle_diff': angle_diff_tmp,
                                                                    'connection': conn,
                                                                    'new_start': new_start,
                                                                    'new_end': new_end,
                                                                    'confidence': 0.0
                                                                }
                                                                candidates = [forced_candidate_auto]
                                                                st.info(f'自動フォールバックで強制候補を作成します（距離={md:.3f}m, 角度差={angle_diff_tmp:.2f}°）')
                                                                try:
                                                                    append_debug(f"Auto-forced candidate applied: pair={w1.get('id')},{w2.get('id')}, min_dist={md}, angle_diff={angle_diff_tmp}")
                                                                except Exception:
                                                                    pass
                                                    except Exception:
                                                        pass

                                                # 強制適用: プレビューで選ばれたペアがある場合、四角形が一致すれば候補が空でも強制的にペアを作成して結合する
                                                if not candidates:
                                                    try:
                                                        last_pair = st.session_state.get('last_preview_pair')
                                                        last_rect = st.session_state.get('last_preview_rect')
                                                        if last_pair:
                                                                    # プレビューで指定されたペアが現在の選択と一致すれば
                                                                    # 四角形の完全一致に依存せず強制適用する（ユーザがプレビューで選択した意図を尊重）
                                                                    id_set = set(last_pair)
                                                                    sel_ids = {selected_walls[0]['id'], selected_walls[1]['id']}
                                                                    if id_set == sel_ids:
                                                                        try:
                                                                            append_debug(f"Preview pair matches selected_walls (ignoring rect): pair={list(id_set)}")
                                                                        except Exception:
                                                                            pass
                                                                        # 最短端点距離と角度差を再計算
                                                                        # （以下は従来の強制適用処理と同じ）
                                                                
                                                                    # 最短端点距離と角度差を再計算
                                                                    endpoints1 = [selected_walls[0]['start'], selected_walls[0]['end']]
                                                                    endpoints2 = [selected_walls[1]['start'], selected_walls[1]['end']]
                                                                    min_dist = None
                                                                    min_pair = None
                                                                    for p1 in endpoints1:
                                                                        for p2 in endpoints2:
                                                                            d = _calc_distance(p1, p2)
                                                                            if min_dist is None or d < min_dist:
                                                                                min_dist = d
                                                                                min_pair = (p1, p2)

                                                                    angle_diff_sel = _calc_angle_diff(selected_walls[0], selected_walls[1])

                                                                    # 接続タイプを最短端点組合せから決定
                                                                    conn = None
                                                                    p1, p2 = min_pair
                                                                    if p1 == selected_walls[0]['end'] and p2 == selected_walls[1]['start']:
                                                                        conn = 'end-start'
                                                                    elif p1 == selected_walls[0]['end'] and p2 == selected_walls[1]['end']:
                                                                        conn = 'end-end'
                                                                    elif p1 == selected_walls[0]['start'] and p2 == selected_walls[1]['start']:
                                                                        conn = 'start-start'
                                                                    elif p1 == selected_walls[0]['start'] and p2 == selected_walls[1]['end']:
                                                                        conn = 'start-end'
                                                                    else:
                                                                        conn = 'end-start'

                                                                    # new_start/new_end を接続タイプに合わせて設定
                                                                    w1 = selected_walls[0]
                                                                    w2 = selected_walls[1]
                                                                    if conn == 'end-start':
                                                                        new_start = w1.get('start')
                                                                        new_end = w2.get('end')
                                                                    elif conn == 'end-end':
                                                                        new_start = w1.get('start')
                                                                        new_end = w2.get('start')
                                                                    elif conn == 'start-start':
                                                                        new_start = w1.get('end')
                                                                        new_end = w2.get('end')
                                                                    elif conn == 'start-end':
                                                                        new_start = w1.get('end')
                                                                        new_end = w2.get('start')
                                                                    else:
                                                                        new_start = None
                                                                        new_end = None

                                                                    forced_candidate = {
                                                                        'wall1': w1,
                                                                        'wall2': w2,
                                                                        'is_chain': False,
                                                                        'distance': min_dist,
                                                                        'angle_diff': angle_diff_sel,
                                                                        'connection': conn,
                                                                        'new_start': new_start,
                                                                        'new_end': new_end,
                                                                        'confidence': 0.0
                                                                    }
                                                                    candidates = [forced_candidate]
                                                                    st.info('プレビュー選択ペアを強制適用して結合を試みます（距離閾値を超えています）。')
                                                                    try:
                                                                        append_debug(f"Forced candidate applied: pair={w1.get('id')},{w2.get('id')}, min_dist={min_dist}, angle_diff={angle_diff_sel}")
                                                                    except Exception:
                                                                        pass
                                                    except Exception:
                                                        pass
                                            
                                                if candidates:
                                                    # 最有力候補の詳細情報を表示（デバッグ用）
                                                    top_candidate = candidates[0]
                                                    st.write(f"**検出されたペア：**")
                                                    if top_candidate.get('is_chain', False):
                                                        chain_wall_ids = [w['id'] for w in top_candidate['walls']]
                                                        st.write(f"チェーン: {chain_wall_ids}")
                                                    else:
                                                        st.write(f"ペア: 壁#{top_candidate['wall1']['id']} + 壁#{top_candidate['wall2']['id']}")
                                                
                                                    # 最有力候補で結合（エラー時は処理を中断して詳細を表示）
                                                    try:
                                                        updated_json = _merge_walls_in_json(updated_json, candidates[:1])
                                                        total_merged_count += 1
                                                        try:
                                                            if top_candidate.get('is_chain'):
                                                                append_debug(f"Merged chain: walls={ [w['id'] for w in top_candidate.get('walls',[])] }")
                                                            else:
                                                                append_debug(f"Merged pair: {top_candidate.get('wall1',{}).get('id')} + {top_candidate.get('wall2',{}).get('id')}, distance={top_candidate.get('distance')}, angle_diff={top_candidate.get('angle_diff')}")
                                                        except Exception:
                                                            pass
                                                    except Exception as e:
                                                        try:
                                                            import traceback
                                                            tb = traceback.format_exc()
                                                        except Exception:
                                                            tb = str(e)
                                                        st.error(f"結合実行中にエラーが発生しました: {e}")
                                                        st.error(f"トレースバック:\n{tb}")
                                                        # 確実に処理を中断する（SystemExit を投げて上位の broad except に捕まらないようにする）
                                                        import sys
                                                        sys.exit(1)
                                                
                                                    # 四角形内の他の不要な線分（中間線）を削除
                                                    # 削除対象は、実際に選択・フィルタされた集合 `walls_to_use` を基準とする。
                                                    # ただし、窓追加などで自動生成された壁（source=='window_added'）は削除対象から除外する。
                                                    walls_to_delete = []
                                                    try:
                                                        basis_list = walls_to_use if walls_to_use is not None else walls_in_selection
                                                        # マージ候補がチェーンかペアかで残すIDを決定
                                                        keep_ids = set()
                                                        try:
                                                            top_cand = candidates[0]
                                                            if top_cand.get('is_chain'):
                                                                # チェーンの最初の壁のみ残す（_merge_walls_in_json と整合）
                                                                keep_ids.add(top_cand['walls'][0]['id'])
                                                            else:
                                                                keep_ids.add(top_cand['wall1']['id'])
                                                        except Exception:
                                                            # 候補情報が見つからない場合は選択2本を残す
                                                            keep_ids.add(selected_walls[0]['id'])
                                                            keep_ids.add(selected_walls[1]['id'])

                                                        for wall in basis_list:
                                                            # 窓追加で生成された壁は保護する
                                                            try:
                                                                if wall.get('source') == 'window_added':
                                                                    try:
                                                                        append_debug(f"Protecting window-added wall from deletion: {wall.get('id')}")
                                                                    except Exception:
                                                                        pass
                                                                    continue
                                                            except Exception:
                                                                pass
                                                            if wall['id'] not in keep_ids:
                                                                walls_to_delete.append(wall['id'])
                                                    except Exception:
                                                        # フォールバック: 以前の挙動に一致させるが、窓追加で生成された壁は削除しない
                                                        for wall in walls_in_selection:
                                                            try:
                                                                if wall.get('source') == 'window_added':
                                                                    continue
                                                            except Exception:
                                                                pass
                                                            if wall['id'] not in [selected_walls[0]['id'], selected_walls[1]['id']]:
                                                                walls_to_delete.append(wall['id'])
                                                
                                                    if walls_to_delete:
                                                        st.write(f"**削除対象の中間線:** 壁#{walls_to_delete}")
                                                        updated_json = _delete_walls_in_json(updated_json, walls_to_delete)
                                                
                                                    color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][rect_idx % 6]
                                                
                                                    # 結合詳細を記録
                                                    merge_details.append({
                                                        'rect_idx': rect_idx,
                                                        'color_name': color_name,
                                                        'is_chain': False,
                                                        'walls': [selected_walls[0]['id'], selected_walls[1]['id']],
                                                        'distance': top_candidate['distance'],
                                                        'direction': direction,
                                                        'deleted_walls': walls_to_delete
                                                    })
                                                else:
                                                    st.warning(f"⚠️ 四角形内の壁が接続されていません")
                                    
                                        if total_merged_count > 0:
                                            st.success(f"✅ 合計 {total_merged_count} 個の選択範囲で結合が完了しました")
                                        
                                            # 結合詳細を表示
                                            st.markdown("**結合結果:**")
                                            for detail in merge_details:
                                                result_text = (
                                                    f"#{detail['rect_idx']+1}（{detail['color_name']}）: "
                                                    f"壁({detail['walls'][0]}) ↔ 壁({detail['walls'][1]}) "
                                                    f"({detail['direction']}) - "
                                                    f"距離: {detail['distance']:.3f}m"
                                                )
                                                if detail.get('deleted_walls'):
                                                    deleted_display = ", ".join([f"壁({wid})" for wid in detail['deleted_walls']])
                                                    result_text += f" | 削除: {deleted_display}"
                                                st.write(result_text)
                                        else:
                                            st.warning("⚠️ 選択範囲内に結合可能な壁線が見つかりません")
                                
                                    elif edit_mode == "線を追加":
                                        # ===== 線を追加モード =====
                                        total_added_count = 0
                                        add_details = []
                                    
                                        for rect_idx, (p1, p2) in enumerate(target_rects):
                                            rect = {
                                                'left': min(p1[0], p2[0]),
                                                'top': min(p1[1], p2[1]),
                                                'width': abs(p2[0] - p1[0]),
                                                'height': abs(p2[1] - p1[1])
                                            }
                                        
                                            # 選択範囲内の壁線を抽出
                                            walls_in_selection = [
                                                wall for wall in updated_json['walls']
                                                if _wall_in_rect(wall, rect, scale, margin, img_height, min_x, min_y, max_x, max_y)
                                            ]
                                        
                                            # 最初の壁（wall1）の高さを取得、なければデフォルト高さを使用
                                            wall_height_to_use = None
                                            if len(walls_in_selection) > 0:
                                                wall_height_to_use = walls_in_selection[0].get('height', None)
                                        
                                            # 線を追加（スケールをセッション状態から取得）
                                            updated_json, direction, new_wall = _add_line_to_json(
                                                updated_json, p1, p2, wall_height=wall_height_to_use, scale=st.session_state.viz_scale
                                            )
                                    
                                    elif edit_mode == "線を削除":
                                        # ===== 線を削除モード =====
                                        total_deleted_count = 0
                                        delete_details = []
                                        walls_to_delete = []  # 削除対象の壁IDリスト
                                    
                                        for rect_idx, (p1, p2) in enumerate(target_rects):
                                            rect = {
                                                'left': min(p1[0], p2[0]),
                                                'top': min(p1[1], p2[1]),
                                                'width': abs(p2[0] - p1[0]),
                                                'height': abs(p2[1] - p1[1])
                                            }
                                        
                                            # 四角形内に完全に含まれる壁線を抽出
                                            walls_in_rect = _filter_walls_strictly_in_rect(
                                                updated_json['walls'], rect, scale, margin, img_height, min_x, min_y, max_x, max_y
                                            )
                                        
                                            if walls_in_rect:
                                                # 四角形内の壁をすべて削除対象に追加
                                                color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][rect_idx % 6]
                                                for wall in walls_in_rect:
                                                    walls_to_delete.append(wall['id'])
                                                    delete_details.append({
                                                        'rect_idx': rect_idx,
                                                        'color_name': color_name,
                                                        'wall_id': wall['id']
                                                    })
                                    
                                        if len(walls_to_delete) > 0:
                                            # 壁を削除
                                            updated_json = _delete_walls_in_json(updated_json, walls_to_delete)
                                        else:
                                            st.warning("⚠️ 削除対象の壁が見つかりません")
                                
                                    elif edit_mode == "床を追加":
                                        # ===== 床を追加モード =====
                                        total_floor_count = 0
                                        floor_details = []
                                        
                                        # JSONに floors キーがなければ初期化
                                        if 'floors' not in updated_json:
                                            updated_json['floors'] = []
                                        
                                        for rect_idx, (p1, p2) in enumerate(target_rects):
                                            # 四角形範囲をメートル座標に変換
                                            px_x1, px_y1 = p1
                                            px_x2, px_y2 = p2
                                            
                                            # 画像座標からメートル座標に変換
                                            def px_to_meter(px_x, px_y):
                                                # 画像座標からメートル座標への変換
                                                meter_x = min_x + (px_x - margin) / scale
                                                meter_y = min_y + (img_height - px_y - margin) / scale
                                                return meter_x, meter_y
                                            
                                            m_x1, m_y1 = px_to_meter(px_x1, px_y1)
                                            m_x2, m_y2 = px_to_meter(px_x2, px_y2)
                                            
                                            # 座標を正規化（x1 < x2, y1 < y2）
                                            floor_x1 = min(m_x1, m_x2)
                                            floor_x2 = max(m_x1, m_x2)
                                            floor_y1 = min(m_y1, m_y2)
                                            floor_y2 = max(m_y1, m_y2)
                                            
                                            # 床データを追加
                                            floor_data = {
                                                'x1': floor_x1,
                                                'y1': floor_y1,
                                                'x2': floor_x2,
                                                'y2': floor_y2
                                            }
                                            updated_json['floors'].append(floor_data)
                                            total_floor_count += 1
                                            
                                            color_name = ["赤", "緑", "青", "黄", "マゼンタ", "シアン"][rect_idx % 6]
                                            floor_details.append({
                                                'rect_idx': rect_idx,
                                                'color_name': color_name,
                                                'x1': floor_x1,
                                                'y1': floor_y1,
                                                'x2': floor_x2,
                                                'y2': floor_y2,
                                                'width': floor_x2 - floor_x1,
                                                'depth': floor_y2 - floor_y1
                                            })
                                        
                                        if total_floor_count > 0:
                                            st.success(f"✅ 合計 {total_floor_count} 個の床を追加しました")
                                            
                                            # 追加詳細を表示
                                            st.markdown("**追加結果:**")
                                            for detail in floor_details:
                                                st.write(
                                                    f"#{detail['rect_idx']+1}（{detail['color_name']}）: "
                                                    f"幅 {detail['width']:.2f}m × 奥行き {detail['depth']:.2f}m"
                                                )
                                        else:
                                            st.warning("⚠️ 床の追加に失敗しました")
                                
                                    # 一時ファイルに保存
                                    temp_json_path = Path(st.session_state.out_dir) / "walls_3d_edited.json"
                                    with open(temp_json_path, 'w', encoding='utf-8') as f:
                                        json.dump(updated_json, f, indent=2, ensure_ascii=False)
                                    
                                    # 再可視化（元の変換と同じスケールを使用）
                                    # 窓追加モードの場合は追加した壁を赤色で表示
                                    temp_viz_path = Path(st.session_state.out_dir) / "visualization_edited.png"
                                    highlight_ids = added_wall_ids if edit_mode == "窓を追加" else None
                                    visualize_3d_walls(str(temp_json_path), str(temp_viz_path), scale=int(viz_scale), highlight_wall_ids=highlight_ids, wall_color=(0, 0, 0), bg_color=(255, 255, 255))
                                
                                    # 3Dビューア生成
                                    temp_viewer_path = Path(st.session_state.out_dir) / "viewer_3d_edited.html"
                                    _generate_3d_viewer_html(temp_json_path, temp_viewer_path)
                                
                                    # セッション状態を更新（スケール校正で最新図を使用するため）
                                    st.session_state.json_bytes = temp_json_path.read_bytes()
                                    st.session_state.viz_bytes = temp_viz_path.read_bytes()
                                
                                    # 編集後の画像を読み込み
                                    edited_viz_bytes = temp_viz_path.read_bytes()
                                    viewer_html_bytes = temp_viewer_path.read_bytes()
                                
                                    # オブジェクト配置モード、線を結合モード、線を追加モードでは、比較表示をせず即座にセッションへ反映して続行する
                                    if edit_mode in ("オブジェクトを配置", "線を結合", "線を追加", "線を削除"):
                                        try:
                                            # 更新済みJSON/可視化/ビューアは既に生成済みの場合がある
                                            # ここでは最新の temp_* が存在すればそれをセッションへ反映する
                                            st.session_state.json_bytes = temp_json_path.read_bytes()
                                            st.session_state.json_name = temp_json_path.name
                                            st.session_state.viz_bytes = temp_viz_path.read_bytes()
                                            st.session_state.viewer_html_bytes = temp_viewer_path.read_bytes()
                                            st.session_state.viewer_html_name = temp_viewer_path.name

                                            # 状態を完全にクリアして続行
                                            st.session_state.rect_coords = []
                                            st.session_state.rect_coords_list = []
                                            st.session_state.last_click = None
                                            if 'window_execution_params' in st.session_state:
                                                del st.session_state.window_execution_params
                                            if edit_mode == "線を結合":
                                                try:
                                                    st.session_state.last_edit_count = total_merged_count
                                                    st.session_state.last_edit_details = merge_details
                                                    st.success(f"✅ 線を結合しました（{total_merged_count} 件）。比較表示をせず保存しました。")
                                                except Exception:
                                                    st.success("✅ 線を結合しました。比較表示をせず保存しました。")
                                            else:
                                                st.success("✅ オブジェクト配置を保存しました。編集結果を比較表示せず次へ進みます。")
                                            time.sleep(0.3)
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"オブジェクト配置の保存中にエラー: {e}")
                                            import traceback
                                            st.code(traceback.format_exc())
                                    else:
                                        # 編集結果をセッション状態に保存（比較表示用）
                                        if edit_mode == "線を結合":
                                            edit_count = total_merged_count
                                            edit_details = merge_details
                                        elif edit_mode == "線を追加":
                                            edit_count = total_added_count
                                            edit_details = add_details
                                        elif edit_mode == "床を追加":
                                            edit_count = total_floor_count
                                            edit_details = floor_details
                                        else:  # 線を削除
                                            edit_count = total_deleted_count
                                            edit_details = delete_details

                                        # デバッグログを保存
                                        debug_log = st.session_state.get('debug_log', [])
                                        
                                        st.session_state.merge_result = {
                                            'original_viz_bytes': original_viz_bytes,
                                            'edited_viz_bytes': edited_viz_bytes,
                                            'json_data': original_json_data,
                                            'updated_json': updated_json,
                                            'temp_json_path': temp_json_path,
                                            'temp_viz_path': temp_viz_path,
                                            'temp_viewer_path': temp_viewer_path,
                                            'viewer_html_bytes': viewer_html_bytes,
                                            'edit_count': edit_count,
                                            'edit_details': edit_details,
                                            'debug_log': debug_log.copy()  # デバッグログをコピーして保存
                                        }
                                        # 編集状態をリセット
                                        st.session_state.rect_coords = []
                                        st.session_state.rect_coords_list = []
                                        # 窓追加パラメータもクリア
                                        if 'window_execution_params' in st.session_state:
                                            del st.session_state.window_execution_params
                                        st.rerun()
                            
                                except Exception as e:
                                    st.error(f"エラーが発生しました: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
            
            # 手動編集モードの最後：編集済みhtmlと照明付きhtmlのダウンロードボタン
            st.divider()
            st.subheader(" 編集済みファイルのダウンロード")
            
            # 編集済み3DビューアHTML（viewer_html_bytesを常に表示）
            if st.session_state.viewer_html_bytes:
                st.download_button(
                    label=" 編集済み3Dモデルをダウンロード",
                    data=st.session_state.viewer_html_bytes,
                    type="primary",
                    file_name=st.session_state.viewer_html_name,
                    mime="text/html"
                )
            
            # 照明付き3DビューアHTML（配置されている場合のみ）
            if st.session_state.get('viewer_html_lights_bytes'):
                st.download_button(
                    label=" 💡照明付き3Dモデルをダウンロード",
                    data=st.session_state.viewer_html_lights_bytes,
                    file_name=st.session_state.viewer_html_lights_name,
                    mime="text/html"
                )


if __name__ == "__main__":
    main()
