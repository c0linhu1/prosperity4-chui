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

HP_LIMIT = 200
VE_LIMIT = 200
VEV_LIMIT = 300
TAKE_CAP = 60
TTE_DAYS = 5
IV_WINDOW = 20


def _d1(S, K, T, sig):
    return (math.log(S/K) + 0.5*sig*sig*T) / (sig*math.sqrt(T))

def bs_call(S, K, T, sig):
    if T <= 1e-6 or sig <= 0 or S <= 0: return max(S-K, 0.0)
    d1 = _d1(S, K, T, sig)
    return S*_N.cdf(d1) - K*_N.cdf(d1-sig*math.sqrt(T))

def bs_vega(S, K, T, sig):
    if T <= 1e-6 or sig <= 0 or S <= 0: return 0.0
    return S*_N.pdf(_d1(S, K, T, sig))*math.sqrt(T)

def iv_newton(S, K, T, mp, init=0.265):
    sig = init
    for _ in range(5):
        p = bs_call(S, K, T, sig)
        v = bs_vega(S, K, T, sig)
        if v < 1e-10: break
        sig -= (p - mp) / v
        sig = max(0.01, min(sig, 2.0))
    return sig


class Trader:
    def get_mid(self, od):
        bb = max(od.buy_orders.keys()) if od.buy_orders else None
        ba = min(od.sell_orders.keys()) if od.sell_orders else None
        if bb is not None and ba is not None: return (bb+ba)/2, bb, ba
        return None, bb, ba

    def get_wall_mid(self, od):
        bb = max(od.buy_orders.keys()) if od.buy_orders else None
        ba = min(od.sell_orders.keys()) if od.sell_orders else None
        if bb is None or ba is None: return None, bb, ba
        wb = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
        wa = min(od.sell_orders.keys(), key=lambda p: -od.sell_orders[p])
        return (wb+wa)/2, bb, ba

    def mm(self, state, product, fair, bb, ba, limit):
        od = state.order_depths[product]
        orders = []; pos = state.position.get(product, 0)
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < fair and pos < TAKE_CAP:
                    qty = min(-od.sell_orders[price], limit-pos, max(TAKE_CAP-pos, 0))
                    if qty > 0: orders.append(Order(product, price, qty)); pos += qty
        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > fair and pos > -TAKE_CAP:
                    qty = min(od.buy_orders[price], limit+pos, max(TAKE_CAP+pos, 0))
                    if qty > 0: orders.append(Order(product, price, -qty)); pos -= qty
        fi = round(fair)
        if od.sell_orders and fi in od.sell_orders and pos < 0:
            qty = min(-od.sell_orders[fi], -pos)
            if qty > 0: orders.append(Order(product, fi, qty)); pos += qty
        if od.buy_orders and fi in od.buy_orders and pos > 0:
            qty = min(od.buy_orders[fi], pos)
            if qty > 0: orders.append(Order(product, fi, -qty)); pos -= qty
        if bb is not None and ba is not None and ba-bb >= 2:
            bp = bb+1; sp = ba-1
            if ba-bb > 4:
                if bp <= fair and limit-pos > 0: orders.append(Order(product, bp, limit-pos))
                if sp >= fair and limit+pos > 0: orders.append(Order(product, sp, -(limit+pos)))
            else:
                if limit-pos > 0: orders.append(Order(product, bp, limit-pos))
                if limit+pos > 0: orders.append(Order(product, sp, -(limit+pos)))
        return orders

    def run(self, state: TradingState):
        result = {}
        td = {}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except: td = {}

        ts = state.timestamp
        T = max((TTE_DAYS - ts/100/10000)/365, 1e-6)

        # ── HP MM ──
        if "HYDROGEL_PACK" in state.order_depths:
            od = state.order_depths["HYDROGEL_PACK"]
            fair, bb, ba = self.get_wall_mid(od)
            if fair is not None:
                result["HYDROGEL_PACK"] = self.mm(state, "HYDROGEL_PACK", fair, bb, ba, HP_LIMIT)

        # ── VE MM (independent) ──
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            fair, bb, ba = self.get_mid(od)
            if fair is not None:
                result["VELVETFRUIT_EXTRACT"] = self.mm(state, "VELVETFRUIT_EXTRACT", fair, bb, ba, VE_LIMIT)

        # ── VE mid for BS ──
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        ve_mid = None
        if ve_od:
            ve_mid, _, _ = self.get_mid(ve_od)

        if ve_mid is None:
            logger.flush(state, result, 0, json.dumps(td))
            return result, 0, json.dumps(td)

        # ── Wide option MM (4000, 4500) with take cap ──
        for strike in [4000]:
            product = f"VEV_{strike}"
            od = state.order_depths.get(product)
            if not od: continue
            pos = state.position.get(product, 0)
            fair, bb, ba = self.get_wall_mid(od)
            if fair is None: continue

            orders = []
            limit = VEV_LIMIT
            opt_take_cap = 60

            # Take (capped like HP)
            if od.sell_orders:
                for price in sorted(od.sell_orders.keys()):
                    if price < fair and pos < opt_take_cap:
                        qty = min(-od.sell_orders[price], limit - pos, max(opt_take_cap - pos, 0))
                        if qty > 0: orders.append(Order(product, price, qty)); pos += qty
            if od.buy_orders:
                for price in sorted(od.buy_orders.keys(), reverse=True):
                    if price > fair and pos > -opt_take_cap:
                        qty = min(od.buy_orders[price], limit + pos, max(opt_take_cap + pos, 0))
                        if qty > 0: orders.append(Order(product, price, -qty)); pos -= qty

            # Flatten at fair
            fi = round(fair)
            if od.sell_orders and fi in od.sell_orders and pos < 0:
                qty = min(-od.sell_orders[fi], -pos)
                if qty > 0: orders.append(Order(product, fi, qty)); pos += qty
            if od.buy_orders and fi in od.buy_orders and pos > 0:
                qty = min(od.buy_orders[fi], pos)
                if qty > 0: orders.append(Order(product, fi, -qty)); pos -= qty

            # Passive (full size)
            if bb is not None and ba is not None and ba - bb >= 2:
                bp = bb + 1; sp = ba - 1
                if bp <= fair and limit - pos > 0:
                    orders.append(Order(product, bp, limit - pos))
                if sp >= fair and limit + pos > 0:
                    orders.append(Order(product, sp, -(limit + pos)))

            if orders:
                result[product] = orders

        # ── ATM options: rolling IV signal for directional quoting ──
        alpha = 2.0 / (IV_WINDOW + 1)

        for strike in [5000, 5100, 5200, 5300, 5400, 5500]:
            product = f"VEV_{strike}"
            od = state.order_depths.get(product)
            if not od: continue
            mid, bb, ba = self.get_mid(od)
            if mid is None or bb is None or ba is None: continue
            spread = ba - bb

            # Compute IV
            cur_iv = iv_newton(ve_mid, strike, T, mid)

            # Rolling mean IV (EMA)
            key = f"iv_{strike}"
            prev = td.get(key, cur_iv)
            mean_iv = alpha * cur_iv + (1-alpha) * prev
            td[key] = mean_iv

            # Fair price at rolling mean IV
            fair_price = bs_call(ve_mid, strike, T, mean_iv)

            pos = state.position.get(product, 0)
            orders = []

            if spread >= 3:
                # Can post inside the spread. Use signal to pick side.
                if cur_iv > mean_iv + 0.002:
                    # IV above mean: overpriced -> favor selling
                    sell_qty = min(50, VEV_LIMIT + pos)
                    if sell_qty > 0:
                        orders.append(Order(product, ba - 1, -sell_qty))
                    # Also post buy at bb+1 with smaller size (still MM)
                    buy_qty = min(10, VEV_LIMIT - pos)
                    if buy_qty > 0:
                        orders.append(Order(product, bb + 1, buy_qty))

                elif cur_iv < mean_iv - 0.002:
                    # IV below mean: underpriced -> favor buying
                    buy_qty = min(50, VEV_LIMIT - pos)
                    if buy_qty > 0:
                        orders.append(Order(product, bb + 1, buy_qty))
                    sell_qty = min(10, VEV_LIMIT + pos)
                    if sell_qty > 0:
                        orders.append(Order(product, ba - 1, -sell_qty))

                else:
                    # Near mean: balanced MM
                    buy_qty = min(30, VEV_LIMIT - pos)
                    sell_qty = min(30, VEV_LIMIT + pos)
                    if buy_qty > 0:
                        orders.append(Order(product, bb + 1, buy_qty))
                    if sell_qty > 0:
                        orders.append(Order(product, ba - 1, -sell_qty))

            elif spread == 2:
                # bb+1 = mid = ba-1. Only post when signal is clear.
                if cur_iv > mean_iv + 0.003:
                    sell_qty = min(30, VEV_LIMIT + pos)
                    if sell_qty > 0:
                        orders.append(Order(product, ba - 1, -sell_qty))
                elif cur_iv < mean_iv - 0.003:
                    buy_qty = min(30, VEV_LIMIT - pos)
                    if buy_qty > 0:
                        orders.append(Order(product, bb + 1, buy_qty))

            # spread=1: skip

            if orders:
                result[product] = orders

        # ── LONG STRADDLE: buy calls + sell VE to hedge ──
        # Profits from VE moving in EITHER direction
        # Cost: ~600 spread on entry + VE hedge spread
        # Gamma profit if VE moves 50+pts: +1,000-3,000
        STRADDLE_CALLS = {5300: 300, 5400: 300, 5200: 17}
        
        if ve_mid is not None and ve_od is not None:
            import math
            from statistics import NormalDist as _ND
            _n = _ND()
            def _bd(S,K,T,s):
                if T<=1e-6 or s<=0: return 1.0 if S>K else 0.0
                d1=(math.log(S/K)+0.5*s*s*T)/(s*math.sqrt(T))
                return _n.cdf(d1)
            
            T_val = max((TTE_DAYS - ts/100/10000)/365, 1e-6)
            
            # Buy calls gradually
            for strike, target in STRADDLE_CALLS.items():
                product = f"VEV_{strike}"
                od = state.order_depths.get(product)
                if not od: continue
                pos = state.position.get(product, 0)
                remaining = target - pos
                if remaining > 0 and od.sell_orders:
                    orders = result.get(product, [])
                    buy_per_tick = min(20, remaining)
                    for price in sorted(od.sell_orders.keys()):
                        if buy_per_tick <= 0: break
                        qty = min(-od.sell_orders[price], buy_per_tick, VEV_LIMIT - pos)
                        if qty > 0:
                            orders.append(Order(product, price, qty))
                            buy_per_tick -= qty; pos += qty
                    if orders: result[product] = orders

            # Delta hedge: sell VE to offset call delta
            agg_delta = 0
            for strike in list(STRADDLE_CALLS.keys()) + [4000]:
                product = f"VEV_{strike}"
                opt_pos = state.position.get(product, 0)
                if opt_pos != 0:
                    agg_delta += opt_pos * _bd(ve_mid, strike, T_val, 0.265)

            target_ve = max(-VE_LIMIT, min(VE_LIMIT, round(-agg_delta)))
            ve_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            ve_needed = target_ve - ve_pos

            # Only rebalance when off by 25+ (reduce hedge cost)
            if abs(ve_needed) > 25:
                ve_orders = []
                if ve_needed > 0 and ve_od.sell_orders:
                    left = ve_needed
                    for price in sorted(ve_od.sell_orders.keys()):
                        if left <= 0: break
                        qty = min(-ve_od.sell_orders[price], left, VE_LIMIT - ve_pos)
                        if qty > 0:
                            ve_orders.append(Order("VELVETFRUIT_EXTRACT", price, qty))
                            ve_pos += qty; left -= qty
                elif ve_needed < 0 and ve_od.buy_orders:
                    left = -ve_needed
                    for price in sorted(ve_od.buy_orders.keys(), reverse=True):
                        if left <= 0: break
                        qty = min(ve_od.buy_orders[price], left, VE_LIMIT + ve_pos)
                        if qty > 0:
                            ve_orders.append(Order("VELVETFRUIT_EXTRACT", price, -qty))
                            ve_pos -= qty; left -= qty
                if ve_orders:
                    result["VELVETFRUIT_EXTRACT"] = ve_orders

        conversions = 0
        try: trader_data = json.dumps(td)
        except: trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data