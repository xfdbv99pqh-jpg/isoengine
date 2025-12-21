"""
Synthetic Topological Agent (STA) v9.0 - THE FULL PHYSICS ENGINE
FIXED VERSION - With proper error handling for debugging
"""

import time
import json
import logging
import sys
import os
import threading
import traceback
import requests
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional

# =============================================================================
# GPU CONFIGURATION - AUTO-DETECT CUDA
# =============================================================================
try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    xp = cp
    GPU_ENABLED = True
    try:
        GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
    except:
        GPU_NAME = "CUDA GPU"
except (ImportError, Exception) as e:
    xp = np
    GPU_ENABLED = False
    GPU_NAME = "CPU (NumPy)"

# =============================================================================
# WEBSOCKET IMPORT WITH FALLBACK
# =============================================================================
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("[ERROR] websocket-client not installed. Run: pip install websocket-client")

# rel is optional - we can use a simpler approach without it
try:
    import rel
    REL_AVAILABLE = True
except ImportError:
    REL_AVAILABLE = False
    print("[INFO] rel library not found. Using simple WebSocket mode.")

# =============================================================================
# CONFIGURATION
# =============================================================================
STARTING_BALANCE = 10000.0
MIN_VOLUME_USD = 10000000
MAX_VOLUME_USD = 200000000
MIN_TRADES_24H = 5000

# LEVERAGE SETTING
TARGET_LEVERAGE = 5.0

BLACKLIST = ["XBT/USD", "ETH/USD", "SOL/USD", "BTC/USD"]

# TRADING SETTINGS
COOLDOWN_SECONDS = 3.0
ENTRY_THRESHOLD = 0.08  # Sensitivity
AVG_DOWN_THRESHOLD = 0.40

KRAKEN_WS_URL = "wss://ws.kraken.com"
KRAKEN_REST_URL = "https://api.kraken.com/0/public"

# Check for TDA libraries
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("[INFO] ripser not installed. Topology features disabled.")

logging.basicConfig(
    filename='sta_kraken.log',
    level=logging.DEBUG,  # Changed to DEBUG for more info
    format='%(asctime)s %(levelname)s %(message)s'
)

# Also log to console for debugging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
logging.getLogger().addHandler(console_handler)

# =============================================================================
# HELPER FUNCTIONS (SCANNER)
# =============================================================================
def get_kraken_tradeable_pairs():
    try:
        response = requests.get(f"{KRAKEN_REST_URL}/AssetPairs", timeout=10)
        data = response.json()
        if data.get('error'):
            print(f"[ERROR] Kraken API error: {data.get('error')}")
            return {}
        return data.get('result', {})
    except Exception as e:
        print(f"[ERROR] Failed to get tradeable pairs: {e}")
        return {}

def get_kraken_ticker(pairs: List[str]):
    try:
        if not pairs: return {}
        chunk_size = 20
        all_data = {}
        for i in range(0, len(pairs), chunk_size):
            chunk = pairs[i:i+chunk_size]
            pair_str = ",".join(chunk)
            response = requests.get(f"{KRAKEN_REST_URL}/Ticker?pair={pair_str}", timeout=10)
            data = response.json()
            if not data.get('error'):
                all_data.update(data.get('result', {}))
            else:
                print(f"[WARN] Ticker error for chunk: {data.get('error')}")
        return all_data
    except Exception as e:
        print(f"[ERROR] Failed to get ticker: {e}")
        return {}

def kraken_to_ws_pair(pair_name: str) -> str:
    replacements = {'XXBT': 'XBT', 'XETH': 'ETH', 'XXRP': 'XRP', 'XXLM': 'XLM', 'ZUSD': 'USD', 'ZEUR': 'EUR'}
    result = pair_name
    for old, new in replacements.items():
        result = result.replace(old, new)
    if '/' not in result and len(result) >= 6:
        if result.endswith('USD'): result = result[:-3] + '/' + result[-3:]
    return result

def scan_for_target():
    print(f"\n{'='*70}")
    print(f"   KRAKEN SCANNER - HUNTING MID-CAP VOLATILITY")
    print(f"{'='*70}")

    try:
        pairs_data = get_kraken_tradeable_pairs()
        if not pairs_data:
            print("[WARN] No pairs data received, using fallback")
            return "XMR/USD", "XXMRZUSD"

        usd_pairs = []
        for pair_name, pair_info in pairs_data.items():
            if pair_info.get('quote', '') in ['ZUSD', 'USD', 'XXBTZUSD']:
                wsname = pair_info.get('wsname', kraken_to_ws_pair(pair_name))
                if wsname not in BLACKLIST:
                    usd_pairs.append({'rest_name': pair_name, 'ws_name': wsname})

        print(f"[INFO] Found {len(usd_pairs)} USD pairs to scan")

        if not usd_pairs:
            print("[WARN] No USD pairs found, using fallback")
            return "XMR/USD", "XXMRZUSD"

        rest_names = [p['rest_name'] for p in usd_pairs]
        ticker_data = get_kraken_ticker(rest_names)

        if not ticker_data:
            print("[WARN] No ticker data received, using fallback")
            return "XMR/USD", "XXMRZUSD"

        candidates = []
        for pair in usd_pairs:
            rn = pair['rest_name']
            ticker = ticker_data.get(rn)
            if not ticker: continue
            try:
                last = float(ticker['c'][0])
                if last == 0: continue
                vol_usd = float(ticker['v'][1]) * last
                num_trades = int(ticker['t'][1])

                if vol_usd < MIN_VOLUME_USD or vol_usd > MAX_VOLUME_USD: continue
                if num_trades < MIN_TRADES_24H: continue

                high = float(ticker['h'][1])
                low = float(ticker['l'][1])
                volatility = (high - low) / last * 100
                trades_per_min = num_trades / 1440

                score = (volatility * 2.0) + (np.log10(trades_per_min+1) * 10)
                candidates.append({**pair, 'volatility': volatility, 'score': score})
            except Exception as e:
                logging.debug(f"Error processing pair {rn}: {e}")
                continue

        candidates.sort(key=lambda x: x['score'], reverse=True)

        if not candidates:
            print("[WARN] No candidates matched criteria, using fallback")
            return "XMR/USD", "XXMRZUSD"

        top = candidates[0]
        print(f"  >>> TARGET LOCKED: {top['ws_name']} (Vol: {top['volatility']:.1f}%) <<<")
        time.sleep(2)
        return top['ws_name'], top['rest_name']

    except Exception as e:
        print(f"[ERROR] Scanner exception: {e}")
        traceback.print_exc()
        return "XMR/USD", "XXMRZUSD"

# =============================================================================
# FULL PHYSICS ENGINE (Thermodynamics & Topology)
# =============================================================================

@dataclass
class MarketState:
    timestamp: float = 0.0
    temperature: float = 0.0
    entropy: float = 0.0
    regime: str = 'unknown'
    l1_persistence: float = 0.0

@dataclass
class OrderFlowData:
    timestamp: float
    price: float
    volume: float
    side: str

@dataclass
class OrderBookSnapshot:
    timestamp: float
    bids: np.ndarray
    asks: np.ndarray
    @property
    def mid_price(self) -> float:
        if len(self.bids) > 0 and len(self.asks) > 0:
            return (self.bids[0, 0] + self.asks[0, 0]) / 2
        return 0.0
    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        if mid > 0 and len(self.bids) > 0 and len(self.asks) > 0:
            return (self.asks[0, 0] - self.bids[0, 0]) / mid * 100
        return 0.0

@dataclass
class Signal:
    asset: str
    action: str
    direction: float
    metadata: Dict = field(default_factory=dict)

class GPUInformationThermodynamics:
    def __init__(self, window_size=100):
        self.flow_buffer = deque(maxlen=window_size)

    def update(self, flow: OrderFlowData):
        self.flow_buffer.append(flow)

    def compute_temperature_gpu(self, orderbook: OrderBookSnapshot) -> float:
        if len(self.flow_buffer) < 10: return 1.0
        flows = list(self.flow_buffer)
        prices_np = np.array([f.price for f in flows], dtype=np.float32)
        volumes_np = np.array([f.volume for f in flows], dtype=np.float32)

        if GPU_ENABLED:
            import cupy as cp
            prices = cp.asarray(prices_np)
            volumes = cp.asarray(volumes_np)
            velocities = cp.diff(prices) / (prices[:-1] + 1e-9)
            norm_vol = volumes[1:] / (cp.mean(volumes) + 1e-9)
            kinetic = norm_vol * (velocities ** 2)
            T = float(cp.mean(kinetic).get()) * (1.0/(orderbook.spread_pct+0.001)) * 500000
        else:
            velocities = np.diff(prices_np) / (prices_np[:-1] + 1e-9)
            norm_vol = volumes_np[1:] / (np.mean(volumes_np) + 1e-9)
            kinetic = norm_vol * (velocities ** 2)
            T = float(np.mean(kinetic)) * (1.0/(orderbook.spread_pct+0.001)) * 500000
        return float(np.clip(T, 0.1, 100.0))

class GPUSensingLayer:
    def __init__(self, window_size=200):
        self.thermo = GPUInformationThermodynamics(window_size)
        self.price_history = deque(maxlen=window_size)
        self.last_state_time = 0
        self.cached_state = None

    def update(self, flow: OrderFlowData):
        self.thermo.update(flow)
        self.price_history.append(flow.price)

    def compute_state(self, orderbook: OrderBookSnapshot) -> MarketState:
        curr = time.time()
        if self.cached_state and (curr - self.last_state_time) < 0.1:
            return self.cached_state
        self.last_state_time = curr

        temp = self.thermo.compute_temperature_gpu(orderbook)

        l1 = 0.0
        if HAS_RIPSER and len(self.price_history) > 20:
            try:
                prices = list(self.price_history)[-20:]
                embedding = np.column_stack([prices[:-1], prices[1:]])
                dgms = ripser(embedding, maxdim=1)['dgms']
                if len(dgms[1]) > 0:
                    l1 = np.sum(dgms[1][:, 1] - dgms[1][:, 0])
            except Exception as e:
                logging.debug(f"Ripser error: {e}")

        regime = 'high_volatility' if temp > 2.0 else 'normal'
        if l1 > 0.3: regime = 'phase_transition'

        self.cached_state = MarketState(orderbook.timestamp, temp, 0.0, regime, l1)
        return self.cached_state

class GPUActiveInferenceAgent:
    def __init__(self):
        if GPU_ENABLED:
            import cupy as cp
            self.belief = cp.array([0.5, 0.5], dtype=cp.float32)
        else:
            self.belief = np.array([0.5, 0.5], dtype=np.float32)
        self.learning_rate = 0.08

    def infer(self):
        val = self.belief[0] - self.belief[1]
        if GPU_ENABLED:
            import cupy as cp
            return float(cp.tanh(val).get())
        else:
            return float(np.tanh(val))

class GPUSTABot:
    def __init__(self, assets: List[str]):
        self.assets = assets
        self.sensing = {a: GPUSensingLayer() for a in assets}
        self.agent = GPUActiveInferenceAgent()
        self.last_trade_time = {}

    def update(self, asset: str, orderbook: OrderBookSnapshot, flow: Optional[OrderFlowData] = None):
        if flow: self.sensing[asset].update(flow)
        return self.sensing[asset].compute_state(orderbook)

    def generate_signal(self, asset: str, prices: np.ndarray) -> Signal:
        curr = time.time()
        if curr - self.last_trade_time.get(asset, 0) < COOLDOWN_SECONDS:
            return Signal(asset, 'hold', 0, {'temp':0})

        state = self.sensing[asset].cached_state
        if not state: return Signal(asset, 'hold', 0, {})

        prediction = self.agent.infer()

        if len(prices) > 10:
            mom = (prices[-1] - prices[-5]) / prices[-5]
            stability_penalty = 1.0
            if state.l1_persistence > 0.5: stability_penalty = 0.5

            combined = ((prediction * 0.4) + (mom * 100 * 0.6)) * stability_penalty
        else:
            combined = 0

        action = 'hold'
        if combined > ENTRY_THRESHOLD: action = 'buy'
        elif combined < -ENTRY_THRESHOLD: action = 'sell'

        return Signal(asset, action, combined, {'temp': state.temperature, 'l1': state.l1_persistence})

# =============================================================================
# COCKPIT & EXECUTION
# =============================================================================

def clear_screen():
    """Cross-platform screen clear"""
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux/Mac
        os.system('clear')

class KrakenCockpit:
    def __init__(self, ws_pair, rest_pair, bot):
        self.ws_pair = ws_pair
        self.rest_pair = rest_pair
        self.bot = bot

        self.cash = STARTING_BALANCE
        self.shares = 0.0
        self.entry_price = 0.0
        self.realized_pnl = 0.0

        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.last_print = 0
        self.message_count = 0
        self.connected = False

    def get_pnl(self, price):
        if self.shares == 0: return 0.0, 0.0
        if self.shares > 0:
            val = self.shares * price
            cost = self.shares * self.entry_price
            pnl = val - cost
        else:
            val = abs(self.shares) * self.entry_price
            cost_to_close = abs(self.shares) * price
            pnl = val - cost_to_close
            cost = val
        pct = (pnl / cost) * 100 if cost > 0 else 0
        return pnl, pct

    def print_dashboard(self, price, sig_val, temp, l1):
        curr = time.time()
        if curr - self.last_print < 0.25: return
        self.last_print = curr

        clear_screen()

        u_pnl, u_pnl_pct = self.get_pnl(price)

        if self.shares == 0:
            equity = self.cash
            pos_val = 0
        elif self.shares > 0:
            pos_val = self.shares * price
            equity = self.cash + u_pnl
        else:
            pos_val = abs(self.shares) * price
            equity = self.cash + u_pnl

        eff_lev = pos_val / equity if equity > 0 else 0.0

        if self.shares > 0: pos_str = f"LONG {self.shares:.4f}"
        elif self.shares < 0: pos_str = f"SHORT {abs(self.shares):.4f}"
        else: pos_str = "FLAT"

        print(f"")
        print(f" ==================================================")
        print(f"  STA V9.0 PHYSICS ENGINE     GPU: {GPU_NAME}")
        print(f" ==================================================")
        print(f"  TARGET  : {self.ws_pair}")
        print(f"  PRICE   : ${price:,.2f}")
        print(f"  SIGNAL  : {sig_val:+.3f}  [Temp: {temp:.1f} | L1: {l1:.2f}]")
        print(f"  LEVERAGE: {eff_lev:.2f}x  (Target: {TARGET_LEVERAGE}x)")
        print(f" --------------------------------------------------")
        print(f"  EQUITY  : ${equity:,.2f}")
        print(f"  CASH    : ${self.cash:,.2f}")
        print(f" --------------------------------------------------")
        print(f"  POSITION: {pos_str}")
        print(f"  OPEN P&L: ${u_pnl:+.2f} ({u_pnl_pct:+.2f}%)")
        print(f"  BANKED  : ${self.realized_pnl:+.2f}")
        print(f"  TRADES  : {self.total_trades} (W:{self.wins} L:{self.losses})")
        print(f"  MESSAGES: {self.message_count}")
        print(f" ==================================================")
        sys.stdout.flush()

    def execute_trade(self, signal, price):
        if signal.action == 'hold': return

        if self.shares != 0:
            u_pnl, _ = self.get_pnl(price)
            self.realized_pnl += u_pnl
            if self.shares > 0: self.cash += (self.shares * price)
            else: self.cash += (abs(self.shares) * self.entry_price) + u_pnl

            if u_pnl > 0: self.wins += 1
            else: self.losses += 1
            print(f" [EXEC] CLOSED {signal.action.upper()} | P&L: ${u_pnl:.2f}")
            self.shares = 0
            self.total_trades += 1
            time.sleep(2)

        target_size = self.cash * TARGET_LEVERAGE
        if signal.action == 'buy':
            self.shares = target_size / price
            self.entry_price = price
            self.cash -= target_size
            print(f" [EXEC] OPEN LONG @ {price} ({TARGET_LEVERAGE}x)")
        elif signal.action == 'sell':
            self.shares = -(target_size / price)
            self.entry_price = price
            print(f" [EXEC] OPEN SHORT @ {price} ({TARGET_LEVERAGE}x)")

        self.bot.last_trade_time[self.ws_pair] = time.time()
        time.sleep(1)

    def on_open(self, ws):
        print(f"[INFO] WebSocket connected!")
        self.connected = True
        subscribe_msg = json.dumps({
            "event": "subscribe",
            "pair": [self.ws_pair],
            "subscription": {"name": "trade"}
        })
        print(f"[INFO] Subscribing to: {self.ws_pair}")
        ws.send(subscribe_msg)

    def on_error(self, ws, error):
        print(f"[ERROR] WebSocket error: {error}")
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[INFO] WebSocket closed. Status: {close_status_code}, Msg: {close_msg}")
        self.connected = False

    def on_message(self, ws, message):
        self.message_count += 1
        try:
            data = json.loads(message)

            # Handle subscription confirmations and heartbeats
            if isinstance(data, dict):
                event = data.get('event', '')
                if event == 'subscriptionStatus':
                    status = data.get('status', '')
                    print(f"[INFO] Subscription status: {status}")
                    if status == 'error':
                        print(f"[ERROR] Subscription error: {data.get('errorMessage', 'Unknown')}")
                    return
                elif event == 'heartbeat':
                    return  # Ignore heartbeats
                elif event == 'systemStatus':
                    print(f"[INFO] System status: {data.get('status', 'unknown')}")
                    return

            # Handle trade data
            if isinstance(data, list) and len(data) >= 3 and data[-2] == 'trade':
                for t in data[1]:
                    price = float(t[0])
                    vol = float(t[1])

                    flow = OrderFlowData(time.time(), price, vol, 'buy' if t[3]=='b' else 'sell')
                    snapshot = OrderBookSnapshot(time.time(), np.array([[price,1]]), np.array([[price,1]]))

                    self.bot.update(self.ws_pair, snapshot, flow)

                    prices = np.array(self.bot.sensing[self.ws_pair].price_history)
                    sig = self.bot.generate_signal(self.ws_pair, prices)

                    temp = sig.metadata.get('temp', 0.0)
                    l1 = sig.metadata.get('l1', 0.0)
                    self.print_dashboard(price, sig.direction, temp, l1)

                    if sig.action != 'hold':
                        if (self.shares > 0 and sig.action == 'sell') or \
                           (self.shares < 0 and sig.action == 'buy') or \
                           (self.shares == 0):
                            self.execute_trade(sig, price)

        except Exception as e:
            logging.error(f"Message processing error: {e}")
            logging.error(traceback.format_exc())

    def start(self):
        if not WEBSOCKET_AVAILABLE:
            print("[FATAL] websocket-client library not installed!")
            print("Run: pip install websocket-client")
            return

        print(f"\n[INFO] Starting WebSocket connection to {KRAKEN_WS_URL}")
        print(f"[INFO] Target pair: {self.ws_pair}")
        print(f"[INFO] Press Ctrl+C to stop\n")

        websocket.enableTrace(False)  # Set to True for debug output

        ws = websocket.WebSocketApp(
            KRAKEN_WS_URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        # Use rel if available, otherwise use simple run_forever
        if REL_AVAILABLE:
            print("[INFO] Using rel dispatcher")
            ws.run_forever(dispatcher=rel, reconnect=5)
            rel.signal(2, rel.abort)
            rel.dispatch()
        else:
            print("[INFO] Using simple WebSocket mode (no rel)")
            try:
                ws.run_forever(reconnect=5)
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted by user")
            except Exception as e:
                print(f"[ERROR] WebSocket run error: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("  STA KRAKEN TRADING BOT - FIXED VERSION")
    print("=" * 60)
    print(f"GPU Enabled: {GPU_ENABLED} ({GPU_NAME})")
    print(f"WebSocket Available: {WEBSOCKET_AVAILABLE}")
    print(f"REL Available: {REL_AVAILABLE}")
    print(f"Ripser (TDA) Available: {HAS_RIPSER}")
    print("=" * 60)

    try:
        ws_pair, rest_pair = scan_for_target()
        print(f"\n[INFO] Selected pair: {ws_pair} ({rest_pair})")

        bot = GPUSTABot([ws_pair])
        cockpit = KrakenCockpit(ws_pair, rest_pair, bot)
        cockpit.start()

    except KeyboardInterrupt:
        print("\n[INFO] Shutdown requested")
    except Exception as e:
        print(f"\n[FATAL] Unhandled exception: {e}")
        traceback.print_exc()
