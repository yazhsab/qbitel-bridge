from playwright.sync_api import sync_playwright
import os

BASE = os.path.dirname(os.path.abspath(__file__))

with sync_playwright() as p:
    browser = p.chromium.launch()
    for i in range(1, 4):
        html_path = os.path.join(BASE, f"figure{i}.html")
        png_path = os.path.join(BASE, f"figure{i}.png")
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.goto(f"file://{html_path}")
        page.wait_for_timeout(500)
        # Get actual content height
        height = page.evaluate("document.body.scrollHeight")
        page.set_viewport_size({"width": 1200, "height": height + 60})
        page.screenshot(path=png_path, full_page=True)
        print(f"Saved {png_path} ({1200}x{height+60})")
    browser.close()
