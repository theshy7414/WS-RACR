#!/usr/bin/env python3
"""
RealSense 相機診斷工具
檢查 D435i 和 D455 的連接狀態
"""
import pyrealsense2 as rs
import sys

def check_realsense_cameras():
    """檢查所有 RealSense 相機"""
    print("=" * 70)
    print("🔍 RealSense 相機診斷工具")
    print("=" * 70)
    
    # 獲取所有連接的設備
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("\n❌ 沒有找到任何 RealSense 相機")
        print("\n可能原因:")
        print("  1. 相機沒有連接")
        print("  2. USB 連接不穩定")
        print("  3. 權限問題")
        print("\n解決方法:")
        print("  sudo apt install librealsense2-utils")
        print("  realsense-viewer  # 測試相機")
        return False
    
    print(f"\n✅ 找到 {len(devices)} 個 RealSense 設備\n")
    
    d435i_found = False
    d455_found = False
    
    for i, device in enumerate(devices):
        print(f"📷 設備 {i + 1}:")
        print(f"   型號: {device.get_info(rs.camera_info.name)}")
        print(f"   序列號: {device.get_info(rs.camera_info.serial_number)}")
        print(f"   韌體版本: {device.get_info(rs.camera_info.firmware_version)}")
        print(f"   USB 類型: {device.get_info(rs.camera_info.usb_type_descriptor)}")
        
        # 檢查型號
        name = device.get_info(rs.camera_info.name)
        if "D435i" in name or "D435I" in name:
            d435i_found = True
            print("   ✅ 這是 D435i (俯視相機)")
        elif "D455" in name:
            d455_found = True
            print("   ✅ 這是 D455 (正視相機)")
        
        # 檢查感測器
        sensors = device.query_sensors()
        print(f"   感測器數量: {len(sensors)}")
        for j, sensor in enumerate(sensors):
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"      - 感測器 {j + 1}: {sensor_name}")
        
        # 檢查是否被佔用
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device.get_info(rs.camera_info.serial_number))
            profile = pipeline.start(config)
            print("   ✅ 相機可以啟動（未被佔用）")
            pipeline.stop()
        except Exception as e:
            print(f"   ⚠️ 無法啟動相機: {str(e)}")
            print("   💡 可能被其他程式佔用")
        
        print()
    
    # 總結
    print("=" * 70)
    print("📊 檢測總結:")
    print(f"   D435i (俯視): {'✅ 找到' if d435i_found else '❌ 未找到'}")
    print(f"   D455 (正視): {'✅ 找到' if d455_found else '❌ 未找到'}")
    print("=" * 70)
    
    if not d435i_found:
        print("\n⚠️ 警告: 未找到 D435i 相機")
        print("   main_hrc_system.py 需要 D435i 才能運行")
    
    if not d455_found:
        print("\n⚠️ 警告: 未找到 D455 相機")
        print("   系統將缺少手勢識別功能")
    
    return d435i_found and d455_found

def check_camera_processes():
    """檢查是否有其他程式佔用相機"""
    print("\n🔍 檢查相機佔用情況...")
    import subprocess
    
    try:
        # 檢查是否有程式在使用 RealSense
        result = subprocess.run(
            ['lsof', '/dev/video*'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.stdout:
            print("\n⚠️ 以下程式正在使用相機:")
            print(result.stdout)
            print("\n💡 建議:")
            print("   1. 關閉 realsense-viewer")
            print("   2. 檢查是否有其他 Python 程式在運行")
            print("   3. 使用 'pkill -f realsense' 強制關閉")
        else:
            print("✅ 沒有程式佔用相機")
    except FileNotFoundError:
        print("⚠️ lsof 工具未安裝，跳過佔用檢查")
        print("   安裝: sudo apt install lsof")
    except Exception as e:
        print(f"⚠️ 檢查失敗: {e}")

def test_camera_stream(serial_number=None):
    """測試相機串流"""
    print("\n🎬 測試相機串流...")
    
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        if serial_number:
            config.enable_device(serial_number)
            print(f"   使用序列號: {serial_number}")
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        print("   啟動相機...")
        profile = pipeline.start(config)
        
        print("   ✅ 相機串流正常")
        print("   讀取 10 幀測試...")
        
        for i in range(10):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if color_frame:
                print(f"   ✅ 第 {i+1} 幀: {color_frame.get_width()}x{color_frame.get_height()}")
            else:
                print(f"   ❌ 第 {i+1} 幀: 讀取失敗")
        
        pipeline.stop()
        print("   ✅ 串流測試完成")
        return True
        
    except Exception as e:
        print(f"   ❌ 串流測試失敗: {str(e)}")
        return False

if __name__ == "__main__":
    print()
    
    # 檢查相機
    cameras_ok = check_realsense_cameras()
    
    # 檢查佔用情況
    check_camera_processes()
    
    # 如果找到相機，詢問是否測試串流
    if cameras_ok:
        print("\n" + "=" * 70)
        try:
            response = input("是否要測試相機串流? (y/n): ").lower()
            if response == 'y':
                test_camera_stream()
        except KeyboardInterrupt:
            print("\n\n👋 已取消")
    
    print()
    sys.exit(0 if cameras_ok else 1)
