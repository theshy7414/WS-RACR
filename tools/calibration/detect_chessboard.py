#!/usr/bin/env python3
"""
Chessboard Corner Detection
棋盤格角點檢測程式
"""

import cv2
import numpy as np
import sys
from pathlib import Path

def detect_chessboard(image_path, pattern_size=(8, 5)):
    """
    檢測並標示棋盤格角點
    
    Args:
        image_path: 圖片路徑
        pattern_size: 棋盤格內部角點數量 (寬, 高)
    """
    print("=" * 70)
    print("Chessboard Corner Detection")
    print("=" * 70)
    
    # 讀取圖片
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Pattern size: {pattern_size[0]}x{pattern_size[1]} (internal corners)")
    
    # 轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 尋找棋盤格角點
    print("\nSearching for chessboard corners...")
    ret, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
              cv2.CALIB_CB_NORMALIZE_IMAGE +
              cv2.CALIB_CB_FAST_CHECK
    )
    
    if ret:
        print(f"Success! Found {len(corners)} corners")
        
        # 精確化角點位置
        print("Refining corner positions...")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )
        
        # 繪製角點
        result_image = image.copy()
        cv2.drawChessboardCorners(
            result_image,
            pattern_size,
            corners_refined,
            ret
        )
        
        # 顯示結果
        print("\nDisplaying result...")
        print("Press any key to save and exit")
        
        # 調整顯示大小（如果圖片太大）
        display_image = result_image.copy()
        max_width = 1280
        if display_image.shape[1] > max_width:
            scale = max_width / display_image.shape[1]
            new_width = int(display_image.shape[1] * scale)
            new_height = int(display_image.shape[0] * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
        
        cv2.imshow('Chessboard Detection', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 儲存結果
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_chessboard.jpg"
        cv2.imwrite(str(output_path), result_image)
        print(f"\nResult saved to: {output_path}")
        
        # 儲存角點座標
        corners_file = Path(image_path).parent / f"{Path(image_path).stem}_corners.txt"
        with open(corners_file, 'w') as f:
            f.write(f"Chessboard Pattern: {pattern_size[0]}x{pattern_size[1]}\n")
            f.write(f"Total Corners: {len(corners_refined)}\n\n")
            for i, corner in enumerate(corners_refined):
                x, y = corner.ravel()
                f.write(f"Corner {i}: ({x:.2f}, {y:.2f})\n")
        print(f"Corner coordinates saved to: {corners_file}")
        
    else:
        print("\nError: Cannot find chessboard pattern!")
        print("\nTroubleshooting:")
        print("  1. Check if pattern size is correct")
        print("  2. Ensure the chessboard is fully visible")
        print("  3. Check lighting conditions")
        print("  4. Try different pattern sizes:")
        print("     - 8x5 (internal corners)")
        print("     - 7x4")
        print("     - 9x6")
        
        # 顯示原圖
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """主程式"""
    if len(sys.argv) < 2:
        print("Usage: python detect_chessboard.py <image_path> [width] [height]")
        print("\nExample:")
        print("  python detect_chessboard.py image.jpg")
        print("  python detect_chessboard.py image.jpg 8 5")
        print("\nDefault pattern size: 8x5 (internal corners)")
        return
    
    image_path = sys.argv[1]
    
    # 解析棋盤格大小
    if len(sys.argv) >= 4:
        pattern_width = int(sys.argv[2])
        pattern_height = int(sys.argv[3])
        pattern_size = (pattern_width, pattern_height)
    else:
        pattern_size = (8, 5)  # 預設值
    
    # 檢測棋盤格
    detect_chessboard(image_path, pattern_size)

if __name__ == "__main__":
    main()
