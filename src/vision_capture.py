"""
Module: vision_capture.py

Provides a single function, capture_frontend_screenshot_base64(), which launches a headless
Playwright Chromium browser, navigates to the GodDev UI (configurable via GODDEV_UI_URL env var),
and returns a base64-encoded JPEG screenshot of the full page. Designed for use by the Meta CTO
agent's visual self-reflection pipeline. Returns None on any failure so callers can gracefully
fall back to text-only analysis.
"""

import base64
import os
from typing import Optional


def capture_frontend_screenshot_base64() -> Optional[str]:
    """
    Capture a full-page screenshot of the GodDev UI and return it as a base64-encoded JPEG string.

    The function:
    - Reads the target URL from the GODDEV_UI_URL environment variable (default http://localhost:8000).
    - Sets a 1440x900 viewport for consistent capture dimensions.
    - Attempts to wait for 'networkidle' (6s timeout), falling back to 'domcontentloaded' (3s).
    - Captures the full page (full_page=True) so all panels (chat, tabs, cost HUD) are visible.
    - Re-encodes the JPEG at quality=25 if the first attempt exceeds ~250KB.
    - Returns None on any error (missing Playwright, missing browser, network failure, etc.)
      and logs a helpful hint when Playwright or its browser is not installed.

    Returns:
        Optional[str]: Base64-encoded JPEG string, or None if capture failed.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Vision Capture Error: playwright not installed. Run: pip install playwright")
        return None

    ui_url = os.environ.get("GODDEV_UI_URL", "http://localhost:8000")

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as exc:
                # Catch FileNotFoundError or similar when Chromium binary is missing
                print(f"Vision Capture Error: cannot launch Chromium ({exc}). Run: playwright install chromium")
                return None

            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            # Attempt networkidle first, fall back to domcontentloaded
            try:
                page.goto(ui_url, wait_until="networkidle", timeout=6000)
            except Exception:
                try:
                    page.goto(ui_url, wait_until="domcontentloaded", timeout=3000)
                except Exception:
                    # Even domcontentloaded failed — take screenshot anyway
                    pass

            # First capture attempt
            screenshot_bytes = page.screenshot(
                type="jpeg",
                quality=30,
                full_page=True,
                animations="disabled",
            )

            # If the screenshot is too large, re-encode at lower quality
            if len(screenshot_bytes) > 250 * 1024:
                screenshot_bytes = page.screenshot(
                    type="jpeg",
                    quality=25,
                    full_page=True,
                    animations="disabled",
                )

            browser.close()

            return base64.b64encode(screenshot_bytes).decode("utf-8")

    except Exception as exc:
        print(f"Vision Capture Error: {exc}")
        return None