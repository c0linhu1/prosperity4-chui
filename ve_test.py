import json
from datamodel import OrderDepth, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import Any, List

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

VE_LIMIT = 200

class Trader:

    def run(self, state: TradingState):
        result = {}

        td = {}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except: td = {}

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            orders = []

            bb = max(od.buy_orders.keys()) if od.buy_orders else None
            ba = min(od.sell_orders.keys()) if od.sell_orders else None

            if bb is not None and ba is not None:
                mid = (bb + ba) / 2
                spread = ba - bb

                # Log OB state
                logger.print(f"ts={state.timestamp} bb={bb} ba={ba} mid={mid} spread={spread} pos={pos}")

                # Log market trades with counterparty info
                for t in state.market_trades.get("VELVETFRUIT_EXTRACT", []):
                    logger.print(f"  MKT: {t.buyer}->{t.seller} @{t.price} x{t.quantity}")

                # Log own trades from last tick
                for t in state.own_trades.get("VELVETFRUIT_EXTRACT", []):
                    logger.print(f"  OWN: {t.buyer}->{t.seller} @{t.price} x{t.quantity}")

                # ── Strategy: same as round 3 VE (wall-mid MM) ──
                # Wall mid
                if od.buy_orders:
                    wb = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
                else:
                    wb = bb
                if od.sell_orders:
                    wa = min(od.sell_orders.keys(), key=lambda p: -od.sell_orders[p])
                else:
                    wa = ba
                fair = (wb + wa) / 2

                # Take all crosses
                for price in sorted(od.sell_orders.keys()):
                    if price < fair and pos < VE_LIMIT:
                        qty = min(-od.sell_orders[price], VE_LIMIT - pos)
                        if qty > 0:
                            orders.append(Order("VELVETFRUIT_EXTRACT", price, qty))
                            pos += qty

                for price in sorted(od.buy_orders.keys(), reverse=True):
                    if price > fair and pos > -VE_LIMIT:
                        qty = min(od.buy_orders[price], VE_LIMIT + pos)
                        if qty > 0:
                            orders.append(Order("VELVETFRUIT_EXTRACT", price, -qty))
                            pos -= qty

                # Flatten at fair
                fi = round(fair)
                if fi in od.sell_orders and pos < 0:
                    qty = min(-od.sell_orders[fi], -pos)
                    if qty > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", fi, qty))
                        pos += qty
                if fi in od.buy_orders and pos > 0:
                    qty = min(od.buy_orders[fi], pos)
                    if qty > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", fi, -qty))
                        pos -= qty

                # Passive quotes with inventory skew
                if spread >= 2:
                    buy_size = max(0, VE_LIMIT - pos)
                    sell_size = max(0, VE_LIMIT + pos)
                    if buy_size > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", bb + 1, buy_size))
                    if sell_size > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", ba - 1, -sell_size))

            result["VELVETFRUIT_EXTRACT"] = orders

        # Empty orders for all other products
        for product in state.order_depths:
            if product not in result:
                result[product] = []

        conversions = 0
        trader_data = json.dumps(td)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data