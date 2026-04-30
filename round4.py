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
TTE_DAYS = 4
FIXED_IV = 0.211

# Per-strike position caps for cross-arb
# ATM strikes with decent spread: uncapped (300)
# Deep ITM or tight spread: conservative (100)
STRIKE_CAPS = {
    5000: VEV_LIMIT,  # 300 - all uncapped
    5100: VEV_LIMIT,  # 300
    5200: VEV_LIMIT,  # 300
    5300: VEV_LIMIT,  # 300
    5400: VEV_LIMIT,  # 300
    5500: VEV_LIMIT,  # 300
}

def bs_call(S, K, T, sig):
    if T <= 1e-6 or sig <= 0 or S <= 0: return max(S - K, 0.0)
    d1 = (math.log(S/K) + 0.5*sig*sig*T) / (sig*math.sqrt(T))
    return S*_N.cdf(d1) - K*_N.cdf(d1-sig*math.sqrt(T))


class Trader:
    def get_wall_mid(self, od):
        bb = max(od.buy_orders.keys()) if od.buy_orders else None
        ba = min(od.sell_orders.keys()) if od.sell_orders else None
        if bb is None or ba is None: return None, bb, ba
        wb = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
        wa = min(od.sell_orders.keys(), key=lambda p: -od.sell_orders[p])
        return (wb+wa)/2, bb, ba

    def get_mid(self, od):
        bb = max(od.buy_orders.keys()) if od.buy_orders else None
        ba = min(od.sell_orders.keys()) if od.sell_orders else None
        if bb is not None and ba is not None: return (bb+ba)/2, bb, ba
        return None, bb, ba

    def mm(self, state, product, fair, bb, ba, limit):
        """Proven round 3 mm()."""
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
        T = max((TTE_DAYS - ts/1_000_000)/365, 1e-6)

        # ── HP: proven wall-mid MM ──
        if "HYDROGEL_PACK" in state.order_depths:
            od = state.order_depths["HYDROGEL_PACK"]
            fair, bb, ba = self.get_wall_mid(od)
            if fair is not None:
                result["HYDROGEL_PACK"] = self.mm(state, "HYDROGEL_PACK", fair, bb, ba, HP_LIMIT)

        # ── VE: sell-heavy MM (builds short, hedged by VEV cross-arb) ──
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        ve_mid = None
        if ve_od and ve_od.buy_orders and ve_od.sell_orders:
            ve_bb = max(ve_od.buy_orders.keys())
            ve_ba = min(ve_od.sell_orders.keys())
            ve_mid = (ve_bb + ve_ba) / 2
            fair = ve_mid
            pos_ve = state.position.get("VELVETFRUIT_EXTRACT", 0)
            orders_ve = []

            # Standard takes (same as mm())
            if ve_od.sell_orders:
                for price in sorted(ve_od.sell_orders.keys()):
                    if price < fair and pos_ve < TAKE_CAP:
                        qty = min(-ve_od.sell_orders[price], VE_LIMIT-pos_ve, max(TAKE_CAP-pos_ve, 0))
                        if qty > 0: orders_ve.append(Order("VELVETFRUIT_EXTRACT", price, qty)); pos_ve += qty
            if ve_od.buy_orders:
                for price in sorted(ve_od.buy_orders.keys(), reverse=True):
                    if price > fair and pos_ve > -TAKE_CAP:
                        qty = min(ve_od.buy_orders[price], VE_LIMIT+pos_ve, max(TAKE_CAP+pos_ve, 0))
                        if qty > 0: orders_ve.append(Order("VELVETFRUIT_EXTRACT", price, -qty)); pos_ve -= qty

            # Sell-heavy passive: full sells, capped buys
            if ve_ba - ve_bb >= 2:
                bp = ve_bb + 1; sp = ve_ba - 1
                sell_size = max(0, VE_LIMIT + pos_ve)
                buy_size = 0  # no buys, max short bias

                if ve_ba - ve_bb > 4:
                    if bp <= fair and buy_size > 0:
                        orders_ve.append(Order("VELVETFRUIT_EXTRACT", bp, buy_size))
                    if sp >= fair and sell_size > 0:
                        orders_ve.append(Order("VELVETFRUIT_EXTRACT", sp, -sell_size))
                else:
                    if buy_size > 0:
                        orders_ve.append(Order("VELVETFRUIT_EXTRACT", bp, buy_size))
                    if sell_size > 0:
                        orders_ve.append(Order("VELVETFRUIT_EXTRACT", sp, -sell_size))

            result["VELVETFRUIT_EXTRACT"] = orders_ve

        # ── VEV 4000/4500: proven wall-mid MM ──
        for strike in [4000, 4500]:
            product = f"VEV_{strike}"
            od = state.order_depths.get(product)
            if not od: continue
            fair, bb, ba = self.get_wall_mid(od)
            if fair is None: continue
            result[product] = self.mm(state, product, fair, bb, ba, VEV_LIMIT)

        # ── VEV 5000-5500: cross-product arb with per-strike caps ──
        if ve_mid and T > 1e-5:
            for K in [5000, 5100, 5200, 5300, 5400, 5500]:
                product = f"VEV_{K}"
                od = state.order_depths.get(product)
                if not od or not od.buy_orders or not od.sell_orders:
                    result[product] = []
                    continue

                pos = state.position.get(product, 0)
                bb = max(od.buy_orders.keys())
                ba = min(od.sell_orders.keys())
                spread = ba - bb
                cap = STRIKE_CAPS.get(K, 100)

                bs_fair = bs_call(ve_mid, K, T, FIXED_IV)
                orders = []

                # Take mispriced: buy when cheap, sell when expensive
                for price in sorted(od.sell_orders.keys()):
                    if price < bs_fair - 0.5 and pos < cap:
                        qty = min(-od.sell_orders[price], cap - pos, 50)
                        if qty > 0:
                            orders.append(Order(product, price, qty))
                            pos += qty

                for price in sorted(od.buy_orders.keys(), reverse=True):
                    if price > bs_fair + 0.5 and pos > -cap:
                        qty = min(od.buy_orders[price], cap + pos, 50)
                        if qty > 0:
                            orders.append(Order(product, price, -qty))
                            pos -= qty

                # Passive around BS fair
                if spread >= 2:
                    buy_price = max(bb + 1, round(bs_fair) - 1)
                    sell_price = min(ba - 1, round(bs_fair) + 1)
                    if sell_price <= buy_price:
                        buy_price = bb + 1; sell_price = ba - 1

                    buy_size = max(0, min(cap - pos, VEV_LIMIT - pos))
                    sell_size = max(0, min(cap + pos, VEV_LIMIT + pos))
                    if buy_size > 0 and buy_price <= bs_fair:
                        orders.append(Order(product, buy_price, buy_size))
                    if sell_size > 0 and sell_price >= bs_fair:
                        orders.append(Order(product, sell_price, -sell_size))

                result[product] = orders

        # Fill empty
        for product in state.order_depths:
            if product not in result:
                result[product] = []

        conversions = 0
        try: trader_data = json.dumps(td)
        except: trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data