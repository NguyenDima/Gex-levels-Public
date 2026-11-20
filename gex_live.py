import time
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

# Optional clipboard support (only works on local Windows/Mac, safely ignored on cloud)
try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

# Optional Playwright support for automated TradingView Cloud Sync
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# ================= TELEGRAM CONFIGURATION =================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
# ==========================================================

NUM_EXPIRATIONS = 3
UPDATE_INTERVAL = 180
RISK_FREE_RATE = 0.045
CHAT_ID_FILE = "tg_chat_id.txt"

MARKETS = {
    "NQ":   {"opt": "QQQ", "fut": "NQ=F",   "band": 0.030},
    "ES":   {"opt": "SPY", "fut": "ES=F",   "band": 0.025},
    "YM":   {"opt": "DIA", "fut": "YM=F",   "band": 0.025},
    "GC":   {"opt": "GLD", "fut": "GC=F",   "band": 0.030},
    "FDAX": {"opt": "SPY", "fut": "^GDAXI", "band": 0.025},
}

def get_chat_id():
    global TELEGRAM_CHAT_ID
    if TELEGRAM_CHAT_ID:
        return TELEGRAM_CHAT_ID
    if os.path.exists(CHAT_ID_FILE):
        with open(CHAT_ID_FILE, "r") as f:
            saved = f.read().strip()
            if saved:
                TELEGRAM_CHAT_ID = saved
                return TELEGRAM_CHAT_ID
    return None

def send_telegram(text):
    token = TELEGRAM_BOT_TOKEN.strip()
    chat_id = get_chat_id()
    if not token or not chat_id:
        print("[Telegram] Token or Chat ID missing.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }

    # Persistent session with automatic backoff retry for ConnectionResetError(104)
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    for attempt in range(1, 4):
        try:
            resp = session.post(url, json=payload, timeout=15).json()
            if resp.get("ok"):
                print("[Telegram SUCCESS] Message delivered!")
                return True
            else:
                print(f"[Telegram API Error] {resp}")
                return False
        except Exception as e:
            print(f"[Telegram Send Attempt {attempt}/3 Error] {e}")
            time.sleep(2 * attempt)
    return False

def calc_bs_gamma(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    eff_sigma = sigma if (sigma is not None and sigma > 0.02 and sigma < 3.0) else 0.18
    d1 = (np.log(S / K) + (r + 0.5 * eff_sigma**2) * T) / (eff_sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * eff_sigma * np.sqrt(T))

def get_latest_price(ticker_obj):
    hist = ticker_obj.history(period="5d", interval="5m")
    if not hist.empty and 'Close' in hist:
        closes = hist['Close'].dropna()
        if not closes.empty:
            return float(closes.iloc[-1])
    hist_d = ticker_obj.history(period="5d")
    return float(hist_d['Close'].dropna().iloc[-1])

def fetch_single_market(mkt_key):
    cfg = MARKETS[mkt_key]
    tk = yf.Ticker(cfg["opt"])
    fut_tk = yf.Ticker(cfg["fut"])

    spot = get_latest_price(tk)
    fut_spot = get_latest_price(fut_tk)
    ratio = fut_spot / spot
    max_band = cfg.get("band", 0.030)

    expirations = tk.options[:NUM_EXPIRATIONS]
    all_chains = []
    now = datetime.now()

    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        days_to_exp = max((exp_date - now).days, 0)
        T = max(days_to_exp, 0.5) / 365.0

        chain = tk.option_chain(exp)
        calls = chain.calls[['strike', 'impliedVolatility', 'openInterest']].copy()
        puts = chain.puts[['strike', 'impliedVolatility', 'openInterest']].copy()
        calls['type'] = 'call'
        puts['type'] = 'put'

        df = pd.concat([calls, puts], ignore_index=True).fillna(0)
        df = df[(df['strike'] >= spot * 0.955) & (df['strike'] <= spot * 1.045)]
        df = df[df['openInterest'] > 0]

        df['gamma'] = df.apply(
            lambda r: calc_bs_gamma(spot, r['strike'], T, RISK_FREE_RATE, r['impliedVolatility']), axis=1
        )
        df['gex'] = df['gamma'] * df['openInterest'] * 100 * (spot ** 2) * 0.01
        df.loc[df['type'] == 'put', 'gex'] *= -1
        df['abs_gex'] = df['gex'].abs()
        all_chains.append(df)

    full_df = pd.concat(all_chains, ignore_index=True)

    strike_gex = full_df.groupby('strike')['gex'].sum().reset_index()
    abs_gex = full_df.groupby('strike')['abs_gex'].sum().reset_index()
    call_gex = full_df[full_df['type'] == 'call'].groupby('strike')['gex'].sum().reset_index()
    put_gex = full_df[full_df['type'] == 'put'].groupby('strike')['gex'].sum().reset_index()

    near_spot = strike_gex[(strike_gex['strike'] >= spot * (1.0 - max_band)) & (strike_gex['strike'] <= spot * (1.0 + max_band))].copy()
    near_spot['sign'] = np.sign(near_spot['gex'].rolling(3, center=True, min_periods=1).mean())
    crossings = near_spot[near_spot['sign'].diff().fillna(0) != 0].copy()

    if not crossings.empty:
        crossings['dist'] = (crossings['strike'] - spot).abs()
        zg_strike = crossings.sort_values('dist').iloc[0]['strike']
    else:
        zg_strike = spot

    zg = zg_strike * ratio

    near_abs = abs_gex[(abs_gex['strike'] >= spot * 0.98) & (abs_gex['strike'] <= spot * 1.02)]
    mag = near_abs.sort_values(by='abs_gex', ascending=False).iloc[0]['strike'] * ratio if not near_abs.empty else spot * ratio

    otm_calls = call_gex[(call_gex['strike'] > spot) & (call_gex['strike'] <= spot * (1.0 + max_band)) & (call_gex['gex'] > 0)].sort_values(by='gex', ascending=False)
    if len(otm_calls) < 4:
        otm_calls = call_gex[call_gex['strike'] > spot].sort_values(by='gex', ascending=False)
    cw = [otm_calls.iloc[min(i, len(otm_calls)-1)]['strike'] * ratio for i in range(4)]

    otm_puts = put_gex[(put_gex['strike'] < spot) & (put_gex['strike'] >= spot * (1.0 - max_band)) & (put_gex['gex'] < 0)].sort_values(by='gex', ascending=True)
    if len(otm_puts) < 4:
        otm_puts = put_gex[put_gex['strike'] < spot].sort_values(by='gex', ascending=True)
    pw = [otm_puts.iloc[min(i, len(otm_puts)-1)]['strike'] * ratio for i in range(4)]

    levels = [zg, mag, cw[0], cw[1], cw[2], cw[3], pw[0], pw[1], pw[2], pw[3]]
    lvl_str = ", ".join([f"{x:.2f}" for x in levels])
    
    tg_block = (
        f"🔹 <b>{mkt_key}</b> (Spot {fut_spot:.1f} | ZG {zg:.1f})\n"
        f"<code>{lvl_str}</code>"
    )
    return lvl_str, tg_block

def build_and_send():
    mkt_strings = {}
    tg_blocks = []
    for mkt in ["NQ", "ES", "YM", "GC", "FDAX"]:
        try:
            lvl_str, block = fetch_single_market(mkt)
            mkt_strings[mkt] = lvl_str
            tg_blocks.append(block)
            print(f"  [OK] {mkt}: {lvl_str}")
        except Exception as e:
            print(f"  [ERR] {mkt}: {e}")
            
    return mkt_strings, "\n\n".join(tg_blocks)

def sync_to_tradingview_cloud(mkt_dict):
    if not HAS_PLAYWRIGHT:
        print("[TV Sync] Playwright not installed, skipping TradingView cloud sync.")
        return

    session_id   = os.environ.get("TV_SESSIONID", "").strip()
    session_sign = os.environ.get("TV_SESSIONID_SIGN", "").strip()
    chart_url    = os.environ.get("TV_CHART_URL", "").strip()

    if not session_id or not chart_url:
        print("[TV Sync] Skipping TradingView sync (TV_SESSIONID or TV_CHART_URL secret not set).")
        return

    print("[TV Sync] Launching headless browser in GitHub Cloud...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})

        cookies = [
            {"name": "sessionid", "value": session_id, "domain": ".tradingview.com", "path": "/"}
        ]
        if session_sign:
            cookies.append({"name": "sessionid_sign", "value": session_sign, "domain": ".tradingview.com", "path": "/"})
        context.add_cookies(cookies)

        page = context.new_page()
        try:
            page.goto(chart_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(15000)

            # 1. Expand collapsed indicator legend if needed
            legend_state = page.evaluate("""() => {
                const findLeaf = () => Array.from(document.querySelectorAll('div')).find(
                    el => el.children.length === 0 && el.textContent && el.textContent.trim() === 'GEX 10L'
                );
                let leaf = findLeaf();
                if (!leaf) return { status: 'not_in_dom' };

                let rect = leaf.getBoundingClientRect();
                if (rect.height === 0) {
                    const toggler = document.querySelector('[data-name="pane-legend-minbtn"], button[class*="toggler"]');
                    if (toggler) {
                        toggler.click();
                        return { status: 'clicked_expander' };
                    }
                }
                return { status: 'already_expanded', height: rect.height };
            }""")
            print(f"[TV Sync] Legend state check: {legend_state}")
            page.wait_for_timeout(1500)

            # 2. Double-click the exact GEX 10L row box to open Settings
            row_info = page.evaluate("""() => {
                const leaf = Array.from(document.querySelectorAll('div')).find(
                    el => el.children.length === 0 && el.textContent && el.textContent.trim() === 'GEX 10L'
                );
                if (!leaf) return null;

                let row = leaf.closest('[data-name="legend-source-item"]');
                if (!row) {
                    let curr = leaf;
                    while (curr && curr !== document.body) {
                        const r = curr.getBoundingClientRect();
                        if (r.width > 40 && r.height >= 14 && r.height <= 40) {
                            row = curr;
                            break;
                        }
                        curr = curr.parentElement;
                    }
                }
                if (!row) return null;

                const rect = row.getBoundingClientRect();
                return { x: rect.left + 25, y: rect.top + (rect.height / 2), width: rect.width, height: rect.height };
            }""")
            print(f"[TV Sync] Exact GEX 10L row box: {row_info}")

            if row_info and row_info["height"] > 0:
                page.mouse.move(row_info["x"], row_info["y"])
                page.wait_for_timeout(600)
                page.mouse.dblclick(row_info["x"], row_info["y"])
                page.wait_for_timeout(1500)

            dialog = page.locator('[data-name="indicator-properties-dialog"]')
            page.wait_for_selector('[data-name="indicator-properties-dialog"]', timeout=10000)
            print("[TV Sync] Settings dialog opened!")

            # 3. Ensure we are on the "Inputs" tab
            inputs_tab = dialog.get_by_text("Inputs", exact=True).first
            if inputs_tab.is_visible():
                inputs_tab.click()
                page.wait_for_timeout(500)

            # 4. Fill the 5 market strings and commit each React input with Enter
            inputs = dialog.locator('input[type="text"], input:not([type])')
            count = inputs.count()
            print(f"[TV Sync] Found {count} text inputs inside GEX 10L dialog.")

            order = ["NQ", "ES", "YM", "GC", "FDAX"]
            for idx, mkt in enumerate(order):
                if mkt in mkt_dict and idx < count:
                    inp = inputs.nth(idx)
                    inp.scroll_into_view_if_needed()
                    inp.click()
                    page.wait_for_timeout(150)
                    page.keyboard.press("Control+A")
                    page.keyboard.press("Backspace")
                    inp.fill(mkt_dict[mkt])
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(200)
                    actual_val = inp.input_value()
                    print(f"[TV Sync] Committed {mkt} -> {actual_val[:22]}...")

            # 5. Click "Ok" button to apply changes to the chart
            ok_btn = dialog.locator('button[name="submit"], button[data-name="submit-button"]').first
            ok_btn.click()
            page.wait_for_timeout(2000)

            # 6. Click the top-right TradingView Cloud Save button AND send Ctrl+S
            saved_click = page.evaluate("""() => {
                const saveBtn = document.querySelector('#header-toolbar-save-load, [data-name="save-load-menu-item"], button[class*="save"]');
                if (saveBtn) {
                    saveBtn.click();
                    return true;
                }
                return false;
            }""")
            page.keyboard.press("Control+s")
            page.wait_for_timeout(5000)
            print(f"[TV Sync] SUCCESS! GEX 10L updated and cloud layout saved (Toolbar click: {saved_click}).")
        except Exception as e:
            print(f"[TV Sync Error] {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    print("=== Multi-Market GEX Engine + Cloud Bot ===")
    mkt_dict, full_tg_message = build_and_send()

    if full_tg_message:
        if HAS_CLIPBOARD:
            try:
                pyperclip.copy(mkt_dict.get("NQ", ""))
            except Exception:
                pass
        
        now_str = datetime.now().strftime("%H:%M:%S")
        tg_msg = (
            f"📊 <b>LIVE GEX LEVELS ({now_str})</b>\n"
            f"<i>Tap any market's numbers below to copy:</i>\n\n"
            f"{full_tg_message}"
        )
        send_telegram(tg_msg)

    if mkt_dict:
        sync_to_tradingview_cloud(mkt_dict)
