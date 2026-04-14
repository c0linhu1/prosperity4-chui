import json
from datamodel import OrderDepth, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import Any, List

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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


class Trader:
    prev_osm_mid = None

    def trade_osmium(self, state: TradingState) -> List[Order]:
        """Mean reverting around 10000 with strong -0.49 lag-1 autocorrelation."""
        product = "ASH_COATED_OSMIUM"
        limit = 80
        fair = 10000
        od = state.order_depths.get(product)
        if not od:
            return []

        orders = []
        pos = state.position.get(product, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else fair - 10
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else fair + 10
        mid = (best_bid + best_ask) / 2

        # Mean reversion signal: if price just went up, lean short. Vice versa.
        signal = 0
        if self.prev_osm_mid is not None:
            diff = mid - self.prev_osm_mid
            if diff > 2:
                signal = -1  # price went up, expect it to come back down
            elif diff < -2:
                signal = 1   # price went down, expect it to come back up
        self.prev_osm_mid = mid

        # Adjust fair value based on signal
        adjusted_fair = fair + signal * 3

        # Take everything below adjusted fair
        for price in sorted(od.sell_orders.keys()):
            if price < adjusted_fair:
                qty = min(-od.sell_orders[price], limit - pos)
                if qty > 0:
                    orders.append(Order(product, price, qty))
                    pos += qty

        # Sell everything above adjusted fair
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > adjusted_fair:
                qty = min(od.buy_orders[price], limit + pos)
                if qty > 0:
                    orders.append(Order(product, price, -qty))
                    pos -= qty

        # Flatten at fair
        if fair in od.sell_orders and pos < 0:
            qty = min(-od.sell_orders[fair], -pos)
            if qty > 0:
                orders.append(Order(product, fair, qty))
                pos += qty
        if fair in od.buy_orders and pos > 0:
            qty = min(od.buy_orders[fair], pos)
            if qty > 0:
                orders.append(Order(product, fair, -qty))
                pos -= qty

        # Passive quotes — skew based on signal
        buy_price = best_bid + 1
        sell_price = best_ask - 1

        if signal == 1:   # expect up, buy more aggressively
            buy_price = best_bid + 2
        elif signal == -1: # expect down, sell more aggressively
            sell_price = best_ask - 2

        if limit - pos > 0:
            orders.append(Order(product, buy_price, limit - pos))
        if limit + pos > 0:
            orders.append(Order(product, sell_price, -(limit + pos)))

        return orders

    def trade_pepper(self, state: TradingState) -> List[Order]:
        """Strong +1/tick drift. Buy and hold max position."""
        product = "INTARIAN_PEPPER_ROOT"
        limit = 80
        od = state.order_depths.get(product)
        if not od:
            return []

        orders = []
        pos = state.position.get(product, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else 0
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else 999999

        # Buy all available asks
        for price in sorted(od.sell_orders.keys()):
            qty = min(-od.sell_orders[price], limit - pos)
            if qty > 0:
                orders.append(Order(product, price, qty))
                pos += qty

        # Post aggressive buy for remaining capacity
        if limit - pos > 0:
            orders.append(Order(product, best_bid + 1, limit - pos))

        return orders

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        product = "EMERALDS"
        limit = 80
        fair = 10000
        od = state.order_depths.get(product)
        if not od:
            return []
        orders = []
        pos = state.position.get(product, 0)
        for price in sorted(od.sell_orders.keys()):
            if price < fair:
                qty = min(-od.sell_orders[price], limit - pos)
                if qty > 0:
                    orders.append(Order(product, price, qty))
                    pos += qty
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > fair:
                qty = min(od.buy_orders[price], limit + pos)
                if qty > 0:
                    orders.append(Order(product, price, -qty))
                    pos -= qty
        if fair in od.sell_orders and pos < 0:
            qty = min(-od.sell_orders[fair], -pos)
            if qty > 0:
                orders.append(Order(product, fair, qty))
                pos += qty
        if fair in od.buy_orders and pos > 0:
            qty = min(od.buy_orders[fair], pos)
            if qty > 0:
                orders.append(Order(product, fair, -qty))
                pos -= qty
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else fair - 10
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else fair + 10
        if limit - pos > 0:
            orders.append(Order(product, best_bid + 1, limit - pos))
        if limit + pos > 0:
            orders.append(Order(product, best_ask - 1, -(limit + pos)))
        return orders

    def trade_tomatoes(self, state: TradingState) -> List[Order]:
        product = "TOMATOES"
        limit = 80
        od = state.order_depths.get(product)
        if not od or not od.buy_orders or not od.sell_orders:
            return []
        orders = []
        pos = state.position.get(product, 0)
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2
        for price in sorted(od.sell_orders.keys()):
            if price < mid:
                qty = min(-od.sell_orders[price], limit - pos)
                if qty > 0:
                    orders.append(Order(product, price, qty))
                    pos += qty
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > mid:
                qty = min(od.buy_orders[price], limit + pos)
                if qty > 0:
                    orders.append(Order(product, price, -qty))
                    pos -= qty
        spread = best_ask - best_bid
        if spread > 2:
            buy_size = min(10, limit - pos)
            sell_size = min(10, limit + pos)
            if buy_size > 0:
                orders.append(Order(product, best_bid + 1, buy_size))
            if sell_size > 0:
                orders.append(Order(product, best_ask - 1, -sell_size))
        return orders

    def run(self, state: TradingState):
        result = {}
        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self.trade_osmium(state)
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result["INTARIAN_PEPPER_ROOT"] = self.trade_pepper(state)
        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.trade_emeralds(state)
        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self.trade_tomatoes(state)
        for product in state.order_depths:
            if product not in result:
                result[product] = []
        conversions = 0
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data