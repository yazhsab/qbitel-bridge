#!/usr/bin/env python3
"""
Convert SVG to PNG using Selenium (headless browser)
This works without needing ImageMagick or Inkscape
"""

from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from PIL import Image
import io
import time

def convert_svg_to_png_selenium(svg_path, png_path, width=1400):
    """Convert SVG to PNG using headless Chrome"""
    try:
        # Set up headless Chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument(f'--window-size={width},2000')

        driver = webdriver.Chrome(options=chrome_options)

        # Load SVG file
        driver.get(f'file:///{svg_path.absolute()}')
        time.sleep(2)  # Wait for rendering

        # Take screenshot
        screenshot = driver.get_screenshot_as_png()
        driver.quit()

        # Save as PNG
        img = Image.open(io.BytesIO(screenshot))
        img.save(png_path, 'PNG')

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    project_root = Path(__file__).parent
    diagrams_dir = project_root / 'diagrams'
    png_dir = diagrams_dir / 'png'
    png_dir.mkdir(exist_ok=True)

    svg_files = list(diagrams_dir.glob('*.svg'))

    print(f"Found {len(svg_files)} SVG files")
    print("Converting to PNG...")

    for svg_file in svg_files:
        if svg_file.name.startswith('.'):
            continue

        png_file = png_dir / f"{svg_file.stem}.png"
        print(f"Converting {svg_file.name}... ", end='')

        if convert_svg_to_png_selenium(svg_file, png_file):
            print("OK")
        else:
            print("FAILED")

if __name__ == "__main__":
    main()
