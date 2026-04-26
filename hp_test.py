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
    def compress_state(self, state, trader_data):
        return [state.timestamp, trader_data, self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades), state.position,
                self.compress_observations(state.observations)]
    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]
    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}
    def compress_trades(self, trades):
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return compressed
    def compress_observations(self, observations):
        co = {}
        for p, o in observations.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [observations.plainValueObservations, co]
    def compress_orders(self, orders):
        compressed = []
        for arr in orders.values():
            for o in arr:
                compressed.append([o.symbol, o.price, o.quantity])
        return compressed
    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
    def truncate(self, value, max_length):
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."

logger = Logger()

LIMIT = 200
INV_LIMIT = 60        # don't let takes push us past ±60
FLATTEN_START = 0.80  # start flattening at 80% through the day
# Day length: 1K ticks in test, 10K in eval. 
# Timestamps go 0 to 99900 (test) or 0 to 999900 (eval).
# We detect which one we're in based on timestamp progression.


class Trader:

    def get_wall_mid(self, od):
        bb = max(od.buy_orders.keys()) if od.buy_orders else None
        ba = min(od.sell_orders.keys()) if od.sell_orders else None
        if bb is None and ba is None:
            return None, None, None
        wb = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p]) if od.buy_orders else bb
        wa = min(od.sell_orders.keys(), key=lambda p: -od.sell_orders[p]) if od.sell_orders else ba
        if wb is not None and wa is not None:
            return (wb + wa) / 2, bb, ba
        elif wb is not None:
            return wb + 8, bb, ba
        else:
            return wa - 8, bb, ba

    def run(self, state: TradingState):
        result = {}
        product = "HYDROGEL_PACK"
        od = state.order_depths.get(product)

        if od:
            orders = []
            pos = state.position.get(product, 0)
            fair, bb, ba = self.get_wall_mid(od)

            if fair is not None:
                fi = round(fair)
                ts = state.timestamp

                # Detect day length: if we see timestamps > 100000, it's 10K ticks
                # For now assume max timestamp = 999900 (10K) or 99900 (1K)
                # Use 999900 as default; the flatten % works either way
                day_end = 999900
                day_progress = ts / day_end  # 0.0 to 1.0
                flattening = day_progress >= FLATTEN_START

                if not flattening:

                    # Phase 1: Take below/above fair
                    # BUT skip takes that push |pos| past INV_LIMIT
                    if od.sell_orders:
                        for price in sorted(od.sell_orders.keys()):
                            if price < fair:
                                # Buy: increases pos. Skip if would exceed +INV_LIMIT
                                if pos >= INV_LIMIT:
                                    continue
                                qty = min(-od.sell_orders[price], LIMIT - pos, INV_LIMIT - pos)
                                if qty > 0:
                                    orders.append(Order(product, price, qty))
                                    pos += qty

                    if od.buy_orders:
                        for price in sorted(od.buy_orders.keys(), reverse=True):
                            if price > fair:
                                # Sell: decreases pos. Skip if would exceed -INV_LIMIT
                                if pos <= -INV_LIMIT:
                                    continue
                                qty = min(od.buy_orders[price], LIMIT + pos, INV_LIMIT + pos)
                                if qty > 0:
                                    orders.append(Order(product, price, -qty))
                                    pos -= qty

                    # Phase 2: Flatten at fair (exact v7)
                    if od.sell_orders and fi in od.sell_orders and pos < 0:
                        qty = min(-od.sell_orders[fi], -pos)
                        if qty > 0:
                            orders.append(Order(product, fi, qty))
                            pos += qty
                    if od.buy_orders and fi in od.buy_orders and pos > 0:
                        qty = min(od.buy_orders[fi], pos)
                        if qty > 0:
                            orders.append(Order(product, fi, -qty))
                            pos -= qty

                    # Phase 3: Passive quotes (exact v7)
                    if bb is not None and ba is not None:
                        spread = ba - bb
                        if spread <= 1:
                            pass
                        elif spread <= 4:
                            buy_qty = LIMIT - pos
                            sell_qty = LIMIT + pos
                            if buy_qty > 0:
                                orders.append(Order(product, bb + 1, buy_qty))
                            if sell_qty > 0:
                                orders.append(Order(product, ba - 1, -sell_qty))
                        else:
                            if LIMIT - pos > 0:
                                bp = bb + 1
                                if bp <= fair:
                                    orders.append(Order(product, bp, LIMIT - pos))
                            if LIMIT + pos > 0:
                                sp = ba - 1
                                if sp >= fair:
                                    orders.append(Order(product, sp, -(LIMIT + pos)))

                else:

                    if pos > 0:
                        # Long: sell to flatten
                        # Take any bids at or above fair
                        if od.buy_orders:
                            for price in sorted(od.buy_orders.keys(), reverse=True):
                                if pos <= 0:
                                    break
                                qty = min(od.buy_orders[price], pos)
                                if qty > 0:
                                    orders.append(Order(product, price, -qty))
                                    pos -= qty
                        # Post passive sell at bb+1
                        if pos > 0 and bb is not None:
                            orders.append(Order(product, bb + 1, -pos))

                    elif pos < 0:
                        # Short: buy to flatten
                        # Take any asks at or below fair
                        if od.sell_orders:
                            for price in sorted(od.sell_orders.keys()):
                                if pos >= 0:
                                    break
                                qty = min(-od.sell_orders[price], -pos)
                                if qty > 0:
                                    orders.append(Order(product, price, qty))
                                    pos += qty
                        # Post passive buy at ba-1
                        if pos < 0 and ba is not None:
                            orders.append(Order(product, ba - 1, -pos))

                    # If flat, stop trading (don't accumulate new positions)

            result[product] = orders

        conversions = 0
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data