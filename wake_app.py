import os
from playwright.sync_api import sync_playwright

URL = os.environ.get("STREAMLIT_APP_URL", "https://nexgentrend-wp.streamlit.app")
WAKE_TEXT = "Yes, get this app back up!"


def try_click(page) -> bool:
    # The wake button can be on the page itself or inside a frame.
    for target in [page] + list(page.frames):
        try:
            btn = target.get_by_role("button", name=WAKE_TEXT)
            if btn.count() > 0:
                btn.first.click()
                return True
        except Exception:
            continue
    return False


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(8_000)

    if try_click(page):
        print("App was asleep - clicked the wake-up button.")
        page.wait_for_timeout(60_000)  # give it time to boot
    else:
        print("No wake-up button found - app looks awake.")

    page.wait_for_timeout(20_000)  # stay connected so it counts as real traffic
    browser.close()