#!/usr/bin/env python3
"""Render the extraction UI headlessly and capture screenshots.

Exists because a frontend that typechecks and builds has still never been
looked at, and layout bugs, unreadable contrast and dead controls do not show
up in `tsc`. Also fails loudly on any console error or failed request, which
is the cheapest way to catch a broken fetch before a demo does.

    python scripts/shoot_ui.py [--ui http://127.0.0.1:5173] [--out ../docs/ui]
"""
from __future__ import annotations

import argparse
import os
import sys

from playwright.sync_api import sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", default="http://127.0.0.1:5173")
    ap.add_argument("--out", default=os.path.join("..", "docs", "ui"))
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    problems: list[str] = []
    shots: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1440, "height": 1000},
            color_scheme=a.theme,
        )
        page.on("console", lambda m: problems.append(f"console.{m.type}: {m.text[:200]}")
                if m.type in ("error", "warning") else None)
        page.on("requestfailed", lambda r: problems.append(f"request failed: {r.url} {r.failure}"))

        def shot(name: str) -> None:
            path = os.path.join(out, f"{name}.png")
            page.screenshot(path=path, full_page=True)
            shots.append(path)
            print(f"  captured {name}")

        # --- landing -------------------------------------------------------
        page.goto(f"{a.ui}/extraction", wait_until="networkidle")
        page.wait_for_timeout(700)
        shot("01-landing")

        # --- a VALIDATED run: the grouped headings ---------------------------
        runs = page.locator(".gx-runs button")
        count = runs.count()
        print(f"  {count} run(s) in the sidebar")
        if count == 0:
            problems.append("no runs listed — the UI cannot reach the API")
        else:
            target = None
            for i in range(count):
                if "validated" in runs.nth(i).inner_text().lower():
                    target = i
                    break
            runs.nth(target if target is not None else 0).click()
            page.wait_for_timeout(1200)
            shot("02-validated-grouped")

            # expand every heading so the grouping is fully visible
            heads = page.locator(".xf-heading")
            for i in range(heads.count()):
                if heads.nth(i).get_attribute("aria-expanded") == "false":
                    heads.nth(i).click()
                    page.wait_for_timeout(120)
            shot("03-all-headings-open")

            # a row's source quote
            rows = page.locator(".xf-row-main:not([disabled])")
            if rows.count():
                rows.first.click()
                page.wait_for_timeout(400)
                shot("04-source-evidence")

            for tab, name in (("Form", "05-form"), ("Audit", "06-audit")):
                btn = page.locator(f".gx-tabs button:has-text('{tab}')")
                if btn.count():
                    btn.first.click()
                    page.wait_for_timeout(600)
                    shot(name)

        # --- a CONFLICTED run: the review queue ------------------------------
        for i in range(runs.count()):
            if "conflicted" in runs.nth(i).inner_text().lower():
                runs.nth(i).click()
                page.wait_for_timeout(1200)
                rev = page.locator(".gx-tabs button:has-text('Review')")
                if rev.count():
                    rev.first.click()
                    page.wait_for_timeout(600)
                shot("07-conflict-review")
                break

        # --- narrow viewport --------------------------------------------------
        page.set_viewport_size({"width": 430, "height": 940})
        page.wait_for_timeout(500)
        shot("08-mobile")

        browser.close()

    print(f"\n{len(shots)} screenshot(s) in {out}")
    if problems:
        print(f"\n{len(problems)} console/network problem(s):")
        for p_ in dict.fromkeys(problems):
            print("   -", p_)
        return 1
    print("no console errors, no failed requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
