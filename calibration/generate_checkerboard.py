#!/usr/bin/env python3
"""
Generate checkerboard pattern for camera calibration
Outputs a PDF/PNG ready for printing
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

def create_checkerboard(rows=9, cols=6, square_size_mm=30, dpi=300):
    """
    Create checkerboard pattern
    
    Args:
        rows: Number of internal corners vertically (kotak hitam-putih - 1)
        cols: Number of internal corners horizontally (kotak hitam-putih - 1)
        square_size_mm: Size of each square in millimeters
        dpi: DPI for printing
    """
    
    # Calculate image size in pixels
    # Add 1 to get actual number of squares
    board_width = (cols + 1) * square_size_mm
    board_height = (rows + 1) * square_size_mm
    
    # Convert mm to inches (for printing)
    width_inches = board_width / 25.4
    height_inches = board_height / 25.4
    
    # Calculate pixel dimensions
    width_px = int(width_inches * dpi)
    height_px = int(height_inches * dpi)
    square_size_px = int(square_size_mm * dpi / 25.4)
    
    # Create checkerboard
    board = np.zeros((height_px, width_px), dtype=np.uint8)
    
    for i in range(rows + 1):
        for j in range(cols + 1):
            if (i + j) % 2 == 0:
                y_start = i * square_size_px
                y_end = (i + 1) * square_size_px
                x_start = j * square_size_px
                x_end = (j + 1) * square_size_px
                board[y_start:y_end, x_start:x_end] = 255
    
    return board, width_inches, height_inches

def save_checkerboard(board, width_inches, height_inches, 
                      rows, cols, square_size_mm, 
                      output_prefix="checkerboard"):
    """
    Save checkerboard as PNG and PDF with instructions
    """
    
    # Save as PNG
    png_file = f"{output_prefix}_{cols+1}x{rows+1}.png"
    cv2.imwrite(png_file, board)
    print(f"✅ PNG saved: {png_file}")
    
    # Create PDF with instructions
    pdf_file = f"{output_prefix}_{cols+1}x{rows+1}.pdf"
    
    with PdfPages(pdf_file) as pdf:
        # Page 1: Checkerboard
        fig = plt.figure(figsize=(width_inches, height_inches))
        ax = plt.subplot(111)
        ax.imshow(board, cmap='gray', interpolation='nearest')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, 
                           wspace=0, hspace=0)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Page 2: Instructions
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.subplot(111)
        ax.axis('off')
        
        instructions = f"""
CAMERA CALIBRATION CHECKERBOARD
================================

Pattern Specifications:
• Grid: {cols+1}×{rows+1} squares
• Square size: {square_size_mm}mm × {square_size_mm}mm
• Internal corners: {cols}×{rows}
• Total size: {(cols+1)*square_size_mm}mm × {(rows+1)*square_size_mm}mm

Printing Instructions:
1. Print this PDF at 100% scale (no scaling/fit to page!)
2. Use WHITE paper (matte preferred, avoid glossy)
3. Check printed square size with ruler
4. Mount on FLAT rigid surface (cardboard/foam board)

Usage for Calibration:
• Use this pattern with: {cols} {rows} parameters
• Example: python calibrate_camera.py video.mp4 {cols} {rows}

Recording Tips:
□ Good lighting (no shadows on pattern)
□ Pattern fills 30-80% of frame
□ Capture from multiple angles
□ Keep pattern flat (not bent)
□ Move slowly, avoid motion blur
□ Record 20-30 seconds minimum
        """
        
        ax.text(0.1, 0.9, instructions, transform=ax.transAxes,
                fontsize=11, fontfamily='monospace',
                verticalalignment='top')
        
        pdf.savefig(fig)
        plt.close()
    
    print(f"✅ PDF saved: {pdf_file}")
    print(f"\n📐 Pattern specs: {cols}×{rows} internal corners")
    print(f"📏 Square size: {square_size_mm}mm")
    print(f"📄 Total size: {(cols+1)*square_size_mm}×{(rows+1)*square_size_mm}mm")

def generate_multiple_sizes():
    """Generate common checkerboard sizes"""
    
    common_patterns = [
        # (cols, rows, square_mm, description)
        (9, 6, 30, "Standard (A4 friendly)"),
        (8, 5, 35, "Medium (Letter size)"),
        (11, 8, 25, "Dense (A3/Tabloid)"),
        (7, 5, 40, "Large squares (Easy detection)"),
        (6, 4, 50, "Extra large (Far calibration)"),
    ]
    
    print("\n🎯 Generating common calibration patterns...\n")
    
    for cols, rows, square_mm, desc in common_patterns:
        print(f"Creating {desc}: {cols}×{rows}, {square_mm}mm squares")
        board, w, h = create_checkerboard(rows, cols, square_mm)
        save_checkerboard(board, w, h, rows, cols, square_mm, 
                         f"pattern_{desc.lower().replace(' ', '_').replace('(', '').replace(')', '')}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate checkerboard calibration pattern')
    parser.add_argument('--cols', type=int, default=9,
                       help='Number of internal corners horizontally (default: 9)')
    parser.add_argument('--rows', type=int, default=6,
                       help='Number of internal corners vertically (default: 6)')
    parser.add_argument('--square', type=int, default=30,
                       help='Square size in mm (default: 30)')
    parser.add_argument('--all', action='store_true',
                       help='Generate all common sizes')
    
    args = parser.parse_args()
    
    if args.all:
        generate_multiple_sizes()
    else:
        board, width, height = create_checkerboard(
            args.rows, args.cols, args.square)
        save_checkerboard(board, width, height, 
                         args.rows, args.cols, args.square)
        
        print("\n✨ Checkerboard generated successfully!")
        print(f"📌 When calibrating, use: python calibrate_camera.py <video> {args.cols} {args.rows}")