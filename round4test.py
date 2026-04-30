import json
import math
from datamodel import OrderDepth, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import Any, List
from statistics import NormalDist

_N = NormalDist()

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end
    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""
    def compress_state(self, state, td):
        return [state.timestamp, td, self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades), state.position,
                self.compress_observations(state.observations)]
    def compress_listings(self, l): return [[x.symbol, x.product, x.denomination] for x in l.values()]
    def compress_order_depths(self, od): return {s: [o.buy_orders, o.sell_orders] for s, o in od.items()}
    def compress_trades(self, t):
        c = []
        for a in t.values():
            for x in a: c.append([x.symbol, x.price, x.quantity, x.buyer, x.seller, x.timestamp])
        return c
    def compress_observations(self, o):
        co = {}
        for p, x in o.conversionObservations.items():
            co[p] = [x.bidPrice, x.askPrice, x.transportFees, x.exportTariff, x.importTariff, x.sugarPrice, x.sunlightIndex]
        return [o.plainValueObservations, co]
    def compress_orders(self, o):
        c = []
        for a in o.values():
            for x in a: c.append([x.symbol, x.price, x.quantity])
        return c
    def to_json(self, v): return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))
    def truncate(self, v, m): return v if len(v) <= m else v[:m-3] + "..."

logger = Logger()

# ── Constants ──
HP_LIMIT = 200
VE_LIMIT = 200
VEV_LIMIT = 300
TTE_DAYS = 4  # Round 4

ALL_VEV = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

def _d1(S, K, T, sig):
    return (math.log(S/K) + 0.5*sig*sig*T) / (sig*math.sqrt(T))

def bs_call(S, K, T, sig):
    if T <= 1e-6 or sig <= 0 or S <= 0: return max(S-K, 0.0)
    d1 = _d1(S, K, T, sig)
    return S*_N.cdf(d1) - K*_N.cdf(d1-sig*math.sqrt(T))

def bs_vega(S, K, T, sig):
    if T <= 1e-6 or sig <= 0 or S <= 0: return 0.0
    return S*_N.pdf(_d1(S, K, T, sig))*math.sqrt(T)

def bs_delta(S, K, T, sig):
    if T <= 1e-6 or sig <= 0 or S <= 0: return 1.0 if S > K else 0.0
    return _N.cdf(_d1(S, K, T, sig))

def iv_newton(S, K, T, mp, init=0.24):
    sig = init
    for _ in range(8):
        p = bs_call(S, K, T, sig)
        v = bs_vega(S, K, T, sig)
        if v < 1e-10: break
        sig -= (p - mp) / v
        sig = max(0.01, min(sig, 3.0))
    return sig


class Trader:

    def _bb_ba(self, od):
        bb = max(od.buy_orders.keys()) if od.buy_orders else None
        ba = min(od.sell_orders.keys()) if od.sell_orders else None
        return bb, ba

    def _mid(self, od):
        bb, ba = self._bb_ba(od)
        if bb is None or ba is None: return None, bb, ba
        return (bb + ba) / 2, bb, ba

    def _wall_mid(self, od):
        bb, ba = self._bb_ba(od)
        if bb is None or ba is None: return None, bb, ba
        wb = max(od.buy_orders, key=lambda p: od.buy_orders[p])
        wa = min(od.sell_orders, key=lambda p: -od.sell_orders[p])
        return (wb + wa) / 2, bb, ba

    def aggressive_take_and_mm(self, state, product, fair, bb, ba, limit):
        """
        Strategy: take ALL volume crossing fair (no cap), post full limit at bb+1/ba-1.
        This is the aggressive strategy the top teams use.
        """
        od = state.order_depths[product]
        orders = []
        pos = state.position.get(product, 0)

        # Take ALL asks below fair
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < fair and pos < limit:
                    qty = min(-od.sell_orders[price], limit - pos)
                    if qty > 0:
                        orders.append(Order(product, price, qty))
                        pos += qty

        # Take ALL bids above fair
        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > fair and pos > -limit:
                    qty = min(od.buy_orders[price], limit + pos)
                    if qty > 0:
                        orders.append(Order(product, price, -qty))
                        pos -= qty

        # Flatten at fair
        fi = round(fair)
        if od.sell_orders and fi in od.sell_orders and pos < 0:
            qty = min(-od.sell_orders[fi], -pos)
            if qty > 0: orders.append(Order(product, fi, qty)); pos += qty
        if od.buy_orders and fi in od.buy_orders and pos > 0:
            qty = min(od.buy_orders[fi], pos)
            if qty > 0: orders.append(Order(product, fi, -qty)); pos -= qty

        # Post FULL remaining at bb+1 / ba-1
        if bb is not None and ba is not None and ba - bb >= 2:
            buy_rem = limit - pos
            sell_rem = limit + pos
            if buy_rem > 0:
                orders.append(Order(product, bb + 1, buy_rem))
            if sell_rem > 0:
                orders.append(Order(product, ba - 1, -sell_rem))

        return orders

    def log_market_trades(self, state, td):
        """Log counterparty behavior for analysis."""
        for product, trades in state.market_trades.items():
            for t in trades:
                key = f"mt_{product}"
                if key not in td:
                    td[key] = []
                # Only keep last 20 trades per product to fit in trader_data
                entry = [t.timestamp, t.buyer, t.seller, t.price, t.quantity]
                td[key].append(entry)
                if len(td[key]) > 20:
                    td[key] = td[key][-20:]

    def analyze_bots(self, state, product):
        """
        Check recent market trades: if a specific bot just traded,
        predict the next move. Returns bias: +1 (price going up), -1 (going down), 0 (neutral).
        """
        trades = state.market_trades.get(product, [])
        if not trades:
            return 0

        # Check the most recent trade
        last = trades[-1]
        buyer = last.buyer
        seller = last.seller

        # From data capsule analysis:
        # HP: Mark 38 = aggressive buyer (lifts asks), Mark 14 = aggressive seller (hits bids)
        # When Mark 38 buys, price often at local high -> sell opportunity
        # When Mark 14 sells, price often at local low -> buy opportunity
        if product == "HYDROGEL_PACK":
            if buyer == "Mark 38":
                return -1  # Mark 38 just bought at ask -> price at local high -> lean short
            if seller == "Mark 14":
                return +1  # Mark 14 just sold at bid -> price at local low -> lean long

        # VE: Mark 55 and Mark 01 trade actively
        if product == "VELVETFRUIT_EXTRACT":
            # When Mark 01 buys from Mark 55, often at a low
            if buyer == "Mark 01":
                return +1
            if seller == "Mark 01":
                return -1

        return 0

    def run(self, state: TradingState):
        result = {}
        td = {}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except: td = {}

        ts = state.timestamp
        T = max((TTE_DAYS - ts / 1_000_000) / 365.0, 1e-6)

        # Log market trades with counterparty info
        # (minimal logging to avoid exceeding trader_data limits)
        bot_counts = td.get("bc", {})
        for product, trades in state.market_trades.items():
            for t in trades:
                key = f"{t.buyer}>{t.seller}"
                if key not in bot_counts:
                    bot_counts[key] = 0
                bot_counts[key] += 1
        td["bc"] = bot_counts

        # ── HYDROGEL_PACK: aggressive MM with bot signal ──
        if "HYDROGEL_PACK" in state.order_depths:
            od = state.order_depths["HYDROGEL_PACK"]
            fair, bb, ba = self._wall_mid(od)
            if fair is not None:
                # Skew fair by bot signal
                bias = self.analyze_bots(state, "HYDROGEL_PACK")
                # If Mark 38 just bought (bias=-1), fair should be higher (sell higher)
                # If Mark 14 just sold (bias=+1), fair should be lower (buy lower)
                fair_adj = fair + bias * 0.5
                result["HYDROGEL_PACK"] = self.aggressive_take_and_mm(
                    state, "HYDROGEL_PACK", fair_adj, bb, ba, HP_LIMIT)

        # ── VELVETFRUIT_EXTRACT: aggressive MM ──
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        ve_mid = None
        if ve_od:
            mid, bb, ba = self._mid(ve_od)
            ve_mid = mid
            if mid is not None:
                bias = self.analyze_bots(state, "VELVETFRUIT_EXTRACT")
                fair_adj = mid + bias * 0.3
                result["VELVETFRUIT_EXTRACT"] = self.aggressive_take_and_mm(
                    state, "VELVETFRUIT_EXTRACT", fair_adj, bb, ba, VE_LIMIT)

        if ve_mid is None:
            for product in state.order_depths:
                if product not in result: result[product] = []
            logger.flush(state, result, 0, json.dumps(td))
            return result, 0, json.dumps(td)

        # ── VEV_4000: deep ITM, treat like underlying ──
        if "VEV_4000" in state.order_depths:
            od = state.order_depths["VEV_4000"]
            fair, bb, ba = self._wall_mid(od)
            if fair is not None:
                result["VEV_4000"] = self.aggressive_take_and_mm(
                    state, "VEV_4000", fair, bb, ba, VEV_LIMIT)

        # ── VEV_4500: also wide spread, MM aggressively ──
        if "VEV_4500" in state.order_depths:
            od = state.order_depths["VEV_4500"]
            fair, bb, ba = self._wall_mid(od)
            if fair is not None:
                result["VEV_4500"] = self.aggressive_take_and_mm(
                    state, "VEV_4500", fair, bb, ba, VEV_LIMIT)

        # ── ATM VEV options: BS fair value + aggressive take ──
        # Key insight from Frankfurt Hedgehogs: fit vol smile, then scalp deviations
        # Simplified version: compute BS fair at rolling mean IV, take all crosses
        alpha = 2.0 / (30 + 1)  # EMA window

        for K in [5000, 5100, 5200, 5300, 5400, 5500]:
            product = f"VEV_{K}"
            od = state.order_depths.get(product)
            if not od: continue
            mid, bb, ba = self._mid(od)
            if mid is None or bb is None or ba is None: continue
            if ba - bb < 1: continue

            # Compute current IV
            try:
                cur_iv = iv_newton(ve_mid, K, T, mid)
            except:
                continue
            if not (0.05 < cur_iv < 3.0): continue

            # Rolling EMA of IV
            key = f"iv_{K}"
            prev_iv = td.get(key, cur_iv)
            mean_iv = alpha * cur_iv + (1 - alpha) * prev_iv
            td[key] = mean_iv

            # BS fair at mean IV
            fair_price = bs_call(ve_mid, K, T, mean_iv)

            # AGGRESSIVE: take ALL volume crossing BS fair
            pos = state.position.get(product, 0)
            orders = []

            if od.sell_orders:
                for price in sorted(od.sell_orders.keys()):
                    if price < fair_price and pos < VEV_LIMIT:
                        qty = min(-od.sell_orders[price], VEV_LIMIT - pos)
                        if qty > 0:
                            orders.append(Order(product, price, qty))
                            pos += qty

            if od.buy_orders:
                for price in sorted(od.buy_orders.keys(), reverse=True):
                    if price > fair_price and pos > -VEV_LIMIT:
                        qty = min(od.buy_orders[price], VEV_LIMIT + pos)
                        if qty > 0:
                            orders.append(Order(product, price, -qty))
                            pos -= qty

            # Post remaining at bb+1/ba-1
            if ba - bb >= 2:
                buy_rem = VEV_LIMIT - pos
                sell_rem = VEV_LIMIT + pos
                if buy_rem > 0:
                    orders.append(Order(product, bb + 1, buy_rem))
                if sell_rem > 0:
                    orders.append(Order(product, ba - 1, -sell_rem))

            if orders:
                result[product] = orders

        # ── VEV_6000 and VEV_6500: deep OTM, spread=1, just post ──
        for K in [6000, 6500]:
            product = f"VEV_{K}"
            od = state.order_depths.get(product)
            if not od: continue
            mid, bb, ba = self._mid(od)
            if mid is None or bb is None or ba is None: continue
            if ba - bb < 2: continue
            pos = state.position.get(product, 0)
            orders = []
            buy_rem = VEV_LIMIT - pos
            sell_rem = VEV_LIMIT + pos
            if buy_rem > 0:
                orders.append(Order(product, bb + 1, buy_rem))
            if sell_rem > 0:
                orders.append(Order(product, ba - 1, -sell_rem))
            if orders:
                result[product] = orders

        # Ensure all products present
        for product in state.order_depths:
            if product not in result:
                result[product] = []

        conversions = 0
        trader_data = json.dumps(td)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data