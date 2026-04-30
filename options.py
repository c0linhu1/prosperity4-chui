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

VEV_LIMIT = 300
ALL_VEV = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
TTE_DAYS = 4

def bs_call(S, K, T, sig):
    if T <= 1e-6 or sig <= 0 or S <= 0: return max(S-K, 0.0)
    d1 = (math.log(S/K) + 0.5*sig*sig*T) / (sig*math.sqrt(T))
    return S*_N.cdf(d1) - K*_N.cdf(d1-sig*math.sqrt(T))

def bs_vega(S, K, T, sig):
    if T <= 1e-6 or sig <= 0 or S <= 0: return 0.0
    d1 = (math.log(S/K) + 0.5*sig*sig*T) / (sig*math.sqrt(T))
    return S*_N.pdf(d1)*math.sqrt(T)

def iv_newton(S, K, T, mp, init=0.24):
    sig = init
    for _ in range(8):
        p = bs_call(S, K, T, sig)
        v = bs_vega(S, K, T, sig)
        if v < 1e-10: break
        sig -= (p-mp)/v
        sig = max(0.01, min(sig, 3.0))
    return sig


class Trader:

    def run(self, state: TradingState):
        result = {}
        td = {}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except: td = {}

        ts = state.timestamp
        T = max((TTE_DAYS - ts / 1_000_000) / 365.0, 1e-6)

        # Get VE mid for BS pricing
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        ve_mid = None
        if ve_od and ve_od.buy_orders and ve_od.sell_orders:
            ve_mid = (max(ve_od.buy_orders.keys()) + min(ve_od.sell_orders.keys())) / 2

        # EMA IV tracking
        alpha = 2.0 / (30 + 1)

        for K in ALL_VEV:
            product = f"VEV_{K}"
            od = state.order_depths.get(product)
            if not od or not od.buy_orders or not od.sell_orders:
                continue

            pos = state.position.get(product, 0)
            orders = []

            bb = max(od.buy_orders.keys())
            ba = min(od.sell_orders.keys())
            mid = (bb + ba) / 2
            spread = ba - bb

            # Compute BS fair if we have VE mid
            bs_fair = None
            if ve_mid and T > 1e-5:
                try:
                    cur_iv = iv_newton(ve_mid, K, T, mid)
                    if 0.05 < cur_iv < 3.0:
                        key = f"iv_{K}"
                        prev_iv = td.get(key, cur_iv)
                        mean_iv = alpha * cur_iv + (1-alpha) * prev_iv
                        td[key] = mean_iv
                        bs_fair = bs_call(ve_mid, K, T, mean_iv)
                except:
                    pass

            # Log state
            logger.print(f"{product} ts={ts} bb={bb} ba={ba} spread={spread} pos={pos} bs={bs_fair:.1f}" if bs_fair else f"{product} ts={ts} bb={bb} ba={ba} spread={spread} pos={pos}")

            for t in state.market_trades.get(product, []):
                logger.print(f"  MKT: {t.buyer}->{t.seller} @{t.price} x{t.quantity}")
            for t in state.own_trades.get(product, []):
                logger.print(f"  OWN: {t.buyer}->{t.seller} @{t.price} x{t.quantity}")

            # === STRATEGY ===
            # Wide-spread products (4000, 4500, 5000, 5100, 5200): wall-mid MM
            # Narrow-spread products (5300-6500): BS fair value taking + aggressive posting

            if spread >= 3:
                # Wide spread: classic MM
                wb = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
                wa = min(od.sell_orders.keys(), key=lambda p: -od.sell_orders[p])
                fair = (wb + wa) / 2

                # Take crosses
                for price in sorted(od.sell_orders.keys()):
                    if price < fair and pos < VEV_LIMIT:
                        qty = min(-od.sell_orders[price], VEV_LIMIT - pos)
                        if qty > 0:
                            orders.append(Order(product, price, qty))
                            pos += qty
                for price in sorted(od.buy_orders.keys(), reverse=True):
                    if price > fair and pos > -VEV_LIMIT:
                        qty = min(od.buy_orders[price], VEV_LIMIT + pos)
                        if qty > 0:
                            orders.append(Order(product, price, -qty))
                            pos -= qty

                # Passive inside spread
                buy_size = max(0, VEV_LIMIT - pos)
                sell_size = max(0, VEV_LIMIT + pos)
                if buy_size > 0:
                    orders.append(Order(product, bb + 1, buy_size))
                if sell_size > 0:
                    orders.append(Order(product, ba - 1, -sell_size))

            elif spread == 2:
                # Spread=2: post at bb+1 (=ba-1) both sides
                # Also use BS fair to decide direction
                post_price = bb + 1  # = ba - 1

                if bs_fair is not None:
                    if bs_fair > post_price + 0.3:
                        # Underpriced, lean long
                        buy_size = min(VEV_LIMIT - pos, VEV_LIMIT)
                        if buy_size > 0:
                            orders.append(Order(product, post_price, buy_size))
                    elif bs_fair < post_price - 0.3:
                        # Overpriced, lean short
                        sell_size = min(VEV_LIMIT + pos, VEV_LIMIT)
                        if sell_size > 0:
                            orders.append(Order(product, post_price, -sell_size))
                    else:
                        # Neutral: both sides
                        buy_size = max(0, VEV_LIMIT - pos)
                        sell_size = max(0, VEV_LIMIT + pos)
                        if buy_size > 0:
                            orders.append(Order(product, post_price, buy_size))
                        if sell_size > 0:
                            orders.append(Order(product, post_price, -sell_size))
                else:
                    buy_size = max(0, VEV_LIMIT - pos)
                    sell_size = max(0, VEV_LIMIT + pos)
                    if buy_size > 0:
                        orders.append(Order(product, post_price, buy_size))
                    if sell_size > 0:
                        orders.append(Order(product, post_price, -sell_size))

            elif spread == 1:
                # Spread=1: can't post inside. Must take or post AT bid/ask.
                # Strategy: use BS fair to decide whether to take
                # If BS says fair > ask: BUY at ask (take the ask)
                # If BS says fair < bid: SELL at bid (hit the bid)
                # Otherwise: post at bid (compete with Mark 01) and at ask

                if bs_fair is not None:
                    if bs_fair > ba + 0.3 and pos < VEV_LIMIT:
                        # Take the ask - underpriced
                        qty = min(-od.sell_orders[ba], VEV_LIMIT - pos)
                        if qty > 0:
                            orders.append(Order(product, ba, qty))
                            pos += qty
                    elif bs_fair < bb - 0.3 and pos > -VEV_LIMIT:
                        # Hit the bid - overpriced
                        qty = min(od.buy_orders[bb], VEV_LIMIT + pos)
                        if qty > 0:
                            orders.append(Order(product, bb, -qty))
                            pos -= qty

                # Always post at bid and ask to compete for passive fills
                buy_size = max(0, VEV_LIMIT - pos)
                sell_size = max(0, VEV_LIMIT + pos)
                if buy_size > 0:
                    orders.append(Order(product, bb, buy_size))
                if sell_size > 0:
                    orders.append(Order(product, ba, -sell_size))

            result[product] = orders

        # Empty for non-VEV
        for product in state.order_depths:
            if product not in result:
                result[product] = []

        conversions = 0
        trader_data = json.dumps(td)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data