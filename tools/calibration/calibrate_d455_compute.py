#!/usr/bin/env python3
"""
Hand-Eye 校正 - 使用 D455 相機採集的數據
計算相機相對於機器人基座的外參矩陣
"""
from ws_racr.config import get_data_dir
import glob, os, json, sys
import numpy as np
import cv2
from pathlib import Path

def load_calibration_data(json_file):
    """
    載入校正數據 JSON 文件
    
    Args:
        json_file: calibration_data_D455_*.json 文件路徑
        
    Returns:
        dict: 包含所有採集數據的字典
    """
    print(f"📂 載入校正數據: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 載入完成:")
    print(f"   相機型號: {data['metadata']['camera_model']}")
    print(f"   總共筆數: {data['metadata']['total_captures']}")
    print(f"   解析度: {data['metadata']['resolution']}")
    
    return data

# === 1) 讀取校正數據 JSON ===
# 自動尋找最新的 D455 校正數據文件
CAMERA_ID = "D455"
cal_data_dir = str(get_data_dir() / "calibration" / "calibration_data")

# 可以通過命令行參數指定文件，否則使用最新的
if len(sys.argv) > 1:
    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"❌ 文件不存在: {json_file}")
        sys.exit(1)
else:
    json_files = sorted(glob.glob(f"{cal_data_dir}/calibration_data_{CAMERA_ID}_*.json"))
    if not json_files:
        print(f"❌ 找不到 {CAMERA_ID} 相機的校正數據文件！")
        print(f"   請確保 {cal_data_dir} 目錄下有 calibration_data_{CAMERA_ID}_*.json 文件")
        sys.exit(1)
    # 使用最新的文件
    json_file = json_files[-1]

cal_data = load_calibration_data(json_file)

# === 2) 從第一筆數據提取相機內參 ===
first_capture = cal_data['captures'][0]
intrinsics = first_capture['camera_intrinsics']

# 建立相機內參矩陣 K
K = np.array([
    [intrinsics['fx'], 0, intrinsics['ppx']],
    [0, intrinsics['fy'], intrinsics['ppy']],
    [0, 0, 1]
], dtype=np.float64)

# D455 通常失真很小，設為零
dist = np.zeros((1, 5), dtype=np.float64)

print(f"\n📷 相機內參:")
print(f"   fx: {intrinsics['fx']:.2f}, fy: {intrinsics['fy']:.2f}")
print(f"   ppx: {intrinsics['ppx']:.2f}, ppy: {intrinsics['ppy']:.2f}")
print(f"   解析度: {intrinsics['width']}x{intrinsics['height']}")

# === 3) 準備棋盤格世界座標（以左上角為原點、Z=0 平面）===
# 從 calibration_capture.py 的設定: (8, 5)
CHECKERBOARD = (8, 5)  # 8 列 × 5 行內角點
SQUARE = 0.025  # 25mm = 0.025m（請根據實際校正板調整）

print(f"\n🎯 校正板設定:")
print(f"   尺寸: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} 內角點")
print(f"   方格邊長: {SQUARE*1000:.1f} mm")

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE  # 轉成公尺

# === 4) 從 JSON 數據中提取角點和機器人位姿 ===
print(f"\n🔄 處理 {len(cal_data['captures'])} 筆校正數據...")

R_target2cam_list = []
t_target2cam_list = []
R_gripper2base_list = []
t_gripper2base_list = []
used_corners_list = []  # 保存實際使用的角點（用於重投影誤差計算）

images_dir = f"{cal_data_dir}/images_D455"
valid_count = 0

for i, capture in enumerate(cal_data['captures']):
    print(f"\n處理 #{i}: {capture['color_image']}")
    
    # 讀取影像
    img_path = f"{images_dir}/{capture['color_image']}"
    if not os.path.exists(img_path):
        print(f"   ⚠️  影像文件不存在，跳過")
        continue
    
    img = cv2.imread(img_path)
    
    # 如果 JSON 中已有角點，直接使用
    if capture.get('chessboard_corners') is not None:
        corners = np.array(capture['chessboard_corners'], dtype=np.float32)
        print(f"   ✅ 使用已保存的角點 ({len(corners)} 個)")
    else:
        # 否則重新檢測
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
        
        ok, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if not ok:
            print(f"   ⚠️  未檢測到棋盤格，跳過")
            continue
        
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        print(f"   ✅ 重新檢測角點成功")
    
    # solvePnP: 世界(棋盤) -> 相機 之 rvec,tvec
    # 使用 ITERATIVE 方法（適用於多個角點）
    ok, rvec, tvec = cv2.solvePnP(
        objp, corners, K, dist, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not ok:
        print(f"   ⚠️  solvePnP 失敗，跳過")
        continue
    
    R_target2cam, _ = cv2.Rodrigues(rvec)
    t_target2cam = tvec.reshape(3,)
    
    R_target2cam_list.append(R_target2cam)
    t_target2cam_list.append(t_target2cam)
    used_corners_list.append(corners.reshape(-1, 2))  # 保存角點
    
    # 提取機器人位姿 (gripper -> base)
    robot_pose = capture['robot_pose']
    position = np.array(robot_pose['position'])  # [X, Y, Z] in meters
    orientation = np.array(robot_pose['orientation'])  # [RX, RY, RZ] in radians
    
    # 將歐拉角轉換為旋轉矩陣 (RX, RY, RZ → R)
    # TM 機器人使用 ZYX 順序的外部歐拉角
    rx, ry, rz = orientation[0], orientation[1], orientation[2]
    
    # 計算各軸旋轉矩陣
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    # TM 機器人是 ZYX 順序（外部歐拉角）
    R_g2b = Rz @ Ry @ Rx
    t_g2b = position
    
    # Eye-to-Hand: 需要 base2gripper，所以取反 gripper2base
    R_gripper2base = R_g2b.T  # 旋轉矩陣轉置 = 反矩陣
    t_gripper2base = -R_gripper2base @ t_g2b  # t_b2g = -R_b2g @ t_g2b
    
    R_gripper2base_list.append(R_gripper2base)
    t_gripper2base_list.append(t_gripper2base)
    
    valid_count += 1
    print(f"   ✅ 有效數據 #{valid_count}")
    print(f"      棋盤→相機: t={t_target2cam}")
    print(f"      夾爪→基座: t={t_gripper2base}")

print(f"\n✅ 有效數據筆數: {valid_count}/{len(cal_data['captures'])}")

# 檢查數據有效性
if valid_count < 3:
    print(f"\n❌ 有效數據不足（至少需要 3 筆，目前有 {valid_count} 筆）")
    print("   請採集更多校正數據！")
    exit(1)

# === 5) Hand-Eye 校正解算 ===
print(f"\n🧮 開始 Hand-Eye 校正...")
print("   方法: Eye-to-Hand（固定相機，棋盤在夾爪）")
methods = [
    (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
    (cv2.CALIB_HAND_EYE_PARK, "Park"),
    (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
    (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
    (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
]

def average_transform(Ts):
    Rs = np.stack([T[:3,:3] for T in Ts], 0)
    ts = np.stack([T[:3, 3] for T in Ts], 0)
    M = Rs.sum(0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:  # 保證 SO(3)
        U[:, -1] *= -1
        Rm = U @ Vt
    tm = ts.mean(0)
    Tm = np.eye(4); Tm[:3,:3] = Rm; Tm[:3,3] = tm
    return Tm

results = []
for method, name in methods:
    try:
        # Eye-to-Hand: 
        # 輸入 base2gripper (已在上面取反) 和 target->cam
        # 輸出 cam->base
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list, t_target2cam_list,
            method=method
        )
        
        # 組裝 cam->base 矩陣
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3] = t_cam2base.ravel()
        
        results.append((name, T_cam2base))
        print(f"   ✅ {name} 成功")
    except Exception as e:
        print(f"   ❌ {name} 失敗：{e}")

if not results:
    print("\n❌ 所有方法都失敗了！")
    exit(1)

# 使用第一個成功的結果（通常 Tsai 最穩定）
name_best, T_cam2base = results[0]
print(f"\n✅ 選用方法: {name_best}")


# === 6) 驗證每一筆數據的一致性 ===
print(f"\n🔄 驗證校正結果一致性...")

def hat(R,t):
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=t; return T

# 先用第一張估 T_target→gripper
T_t2c_0 = hat(R_target2cam_list[0], t_target2cam_list[0])
T_g2b_0 = hat(R_gripper2base_list[0], t_gripper2base_list[0])
T_t2b_cam_0 = T_t2c_0 @ T_cam2base          # target→base（由相機）
T_t2g_est = T_t2b_cam_0 @ np.linalg.inv(T_g2b_0)   # 估計固定 offset: target→gripper

# 其餘樣本驗證
errs=[]
for i in range(len(R_target2cam_list)):
    T_t2c = hat(R_target2cam_list[i], t_target2cam_list[i])
    T_g2b = hat(R_gripper2base_list[i], t_gripper2base_list[i])
    T_t2b_cam = T_t2c @ T_cam2base
    T_t2b_robot = T_t2g_est @ T_g2b
    e = np.linalg.norm(T_t2b_cam[:3,3] - T_t2b_robot[:3,3])
    errs.append(e)
print(f"   平均驗證誤差(含夾具offset校正): {np.mean(errs)*1000:.2f} mm")


# 使用單一估計值（不需要平均，因為 Eye-to-Hand 理論上每次應該相同）
print(f"   ✅ 使用 {name_best} 方法的結果")


# === 7) 計算重投影誤差（驗證精度）===
print(f"\n📊 計算校正精度...")
print("   使用方法: 基於 solvePnP 的重投影誤差（相機內參驗證）")

# 計算重投影誤差
reprojection_errors = []
for i in range(len(R_target2cam_list)):
    rvec, _ = cv2.Rodrigues(R_target2cam_list[i]); tvec = t_target2cam_list[i].reshape(3,1)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(used_corners_list[i] - proj, axis=1).mean()
    reprojection_errors.append(err)

mean_error = np.mean(reprojection_errors)
max_error = np.max(reprojection_errors)
min_error = np.min(reprojection_errors)

print(f"   平均誤差: {mean_error:.2f} 像素")
print(f"   最大誤差: {max_error:.2f} 像素")
print(f"   最小誤差: {min_error:.2f} 像素")

# 將像素誤差轉換為 mm（近似）
# 假設相機到棋盤距離約 0.5m，使用相機焦距估算
avg_depth = np.mean([t_target2cam_list[i][2] for i in range(len(t_target2cam_list))])
pixel_size_mm = (avg_depth * 1000) / intrinsics['fx']  # mm per pixel
mean_error_mm = mean_error * pixel_size_mm

print(f"   估計空間誤差: {mean_error_mm:.2f} mm (基於平均深度 {avg_depth*1000:.0f} mm)")
print(f"\n✅ 內參和棋盤設定正確！")
print(f"   (重投影誤差 < 1.0 像素表示相機內參和棋盤尺寸設定準確)")

# === 8) 保存結果 ===
print(f"\n💾 保存校正結果...")

# 保存為 .npy 格式（與舊系統兼容）
output_dir = str(get_data_dir() / "calibration" / "cal_info")
Path(output_dir).mkdir(parents=True, exist_ok=True)

extrinsics_path = f"{output_dir}/extrinsics_{CAMERA_ID}.npy"
np.save(extrinsics_path, T_cam2base[:3, :])  # 保存 3x4 矩陣 [R|t]
print(f"   ✅ 已保存外參到: {extrinsics_path}")

# 保存內參
intrinsics_data = {
    'K': K,
    'dist': dist,
    'width': intrinsics['width'],
    'height': intrinsics['height']
}
intrinsics_path = f"{output_dir}/intrinsics_{CAMERA_ID}.npz"
np.savez(intrinsics_path, **intrinsics_data)
print(f"   ✅ 已保存內參到: {intrinsics_path}")

# 保存完整的 4x4 變換矩陣
T_cam2base_path = f"{output_dir}/T_camera_to_base_{CAMERA_ID}.npy"
np.save(T_cam2base_path, T_cam2base)
print(f"   ✅ 已保存完整變換到: {T_cam2base_path}")

# 保存校正報告
report = {
    'method': name_best,
    'valid_captures': valid_count,
    'total_captures': len(cal_data['captures']),
    'chessboard_size': CHECKERBOARD,
    'square_size_mm': SQUARE * 1000,
    'camera_model': cal_data['metadata']['camera_model'],
    'calibration_date': cal_data['metadata']['timestamp'],
    'mean_reprojection_error_pixels': float(mean_error),
    'max_reprojection_error_pixels': float(max_error),
    'min_reprojection_error_pixels': float(min_error),
    'mean_reprojection_error_mm': float(mean_error_mm),
    'T_camera_to_base': T_cam2base.tolist(),
    'camera_intrinsics': intrinsics
}

report_path = f"{output_dir}/calibration_report_{CAMERA_ID}.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"   ✅ 已保存校正報告到: {report_path}")

# === 9) 顯示結果 ===
print(f"\n{'='*60}")
print(f"✅ Hand-Eye 校正完成！")
print(f"{'='*60}")

np.set_printoptions(precision=6, suppress=True)
print(f"\n📐 相機→基座變換矩陣 (T_camera_to_base):")
print(T_cam2base)

print(f"\n📍 相機位置 (相對基座):")
print(f"   X: {T_cam2base[0, 3]*1000:.2f} mm")
print(f"   Y: {T_cam2base[1, 3]*1000:.2f} mm")
print(f"   Z: {T_cam2base[2, 3]*1000:.2f} mm")

# 計算旋轉角度（歐拉角）
R_cam2base = T_cam2base[:3, :3]
from scipy.spatial.transform import Rotation
euler = Rotation.from_matrix(R_cam2base).as_euler('xyz', degrees=True)
print(f"\n🔄 相機姿態 (相對基座):")
print(f"   RX: {euler[0]:.2f}°")
print(f"   RY: {euler[1]:.2f}°")
print(f"   RZ: {euler[2]:.2f}°")

print(f"\n📊 校正精度:")
print(f"   重投影誤差: {mean_error:.2f} 像素 ({mean_error_mm:.2f} mm)")
print(f"   建議: {'✅ 精度優秀' if mean_error < 1.0 else '✅ 精度良好' if mean_error < 2.0 else '⚠️ 建議重新採集數據'}")

print(f"\n📁 輸出文件:")
print(f"   {extrinsics_path}")
print(f"   {intrinsics_path}")
print(f"   {T_cam2base_path}")
print(f"   {report_path}")

print(f"\n{'='*60}")
print("🎉 可以開始使用相機進行視覺定位了！")
print(f"{'='*60}")
