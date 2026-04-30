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

HP_LIMIT = 200

class Trader:

    def run(self, state: TradingState):
        result = {}

        td = {}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except: td = {}

        if "HYDROGEL_PACK" in state.order_depths:
            od = state.order_depths["HYDROGEL_PACK"]
            pos = state.position.get("HYDROGEL_PACK", 0)
            orders = []

            bb = max(od.buy_orders.keys()) if od.buy_orders else None
            ba = min(od.sell_orders.keys()) if od.sell_orders else None

            if bb is not None and ba is not None:
                mid = (bb + ba) / 2
                spread = ba - bb

                # Log everything for analysis
                logger.print(f"ts={state.timestamp} bb={bb} ba={ba} mid={mid} spread={spread} pos={pos}")

                # Log market trades with counterparty info
                for t in state.market_trades.get("HYDROGEL_PACK", []):
                    logger.print(f"  MKT: {t.buyer}->{t.seller} @{t.price} x{t.quantity}")

                # Log own trades from last tick
                for t in state.own_trades.get("HYDROGEL_PACK", []):
                    logger.print(f"  OWN: {t.buyer}->{t.seller} @{t.price} x{t.quantity}")

                # Strategy: post at bb+1 and ba-1
                # Inventory-aware: reduce size on the side that grows position
                buy_price = bb + 1
                sell_price = ba - 1

                if spread >= 2:
                    # Inventory skew: if long, post more sells; if short, post more buys
                    # Scale: at pos=0 both sides full; at pos=±limit one side is 0
                    buy_size = max(0, HP_LIMIT - pos)
                    sell_size = max(0, HP_LIMIT + pos)

                    if buy_size > 0:
                        orders.append(Order("HYDROGEL_PACK", buy_price, buy_size))
                    if sell_size > 0:
                        orders.append(Order("HYDROGEL_PACK", sell_price, -sell_size))

            result["HYDROGEL_PACK"] = orders

        # Empty orders for all other products
        for product in state.order_depths:
            if product not in result:
                result[product] = []

        conversions = 0
        trader_data = json.dumps(td)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data