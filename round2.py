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

    def bid(self):
        # Market Access Fee: blind auction, top 50% get extra 25% order flow.
        # Fee is one-time, only paid if accepted.
        # Extra flow is worth ~3-10K XIRECs over the round.
        # Bid enough to be safely above median, but not wasteful.
        return 250

    def trade_osmium(self, state: TradingState) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        limit = 80
        od = state.order_depths.get(product)
        if not od:
            return []

        orders = []
        pos = state.position.get(product, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is None and best_ask is None:
            return []

        # Wall mid fair value: use level with biggest volume on each side
        # The "wall" is the deep liquidity level posted by the market maker bot
        if od.buy_orders:
            wall_bid = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
        else:
            wall_bid = best_bid if best_bid else 9992
        if od.sell_orders:
            wall_ask = min(od.sell_orders.keys(), key=lambda p: -od.sell_orders[p])
        else:
            wall_ask = best_ask if best_ask else 10008
        fair = (wall_bid + wall_ask) / 2

        # Phase 1: Take any orders below/above fair value (immediate edge)
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < fair:
                    qty = min(-od.sell_orders[price], limit - pos)
                    if qty > 0:
                        orders.append(Order(product, price, qty))
                        pos += qty

        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > fair:
                    qty = min(od.buy_orders[price], limit + pos)
                    if qty > 0:
                        orders.append(Order(product, price, -qty))
                        pos -= qty

        # Phase 2: Flatten inventory at fair value if possible
        fair_int = round(fair)
        if od.sell_orders and fair_int in od.sell_orders and pos < 0:
            qty = min(-od.sell_orders[fair_int], -pos)
            if qty > 0:
                orders.append(Order(product, fair_int, qty))
                pos += qty
        if od.buy_orders and fair_int in od.buy_orders and pos > 0:
            qty = min(od.buy_orders[fair_int], pos)
            if qty > 0:
                orders.append(Order(product, fair_int, -qty))
                pos -= qty

        # Phase 3: Passive quotes to capture future taker flow
        # Quote at best+1 to be top of book when takers arrive after us
        if best_bid is not None and limit - pos > 0:
            buy_price = best_bid + 1
            # Don't buy above fair (negative edge)
            if buy_price < fair:
                orders.append(Order(product, buy_price, limit - pos))
            elif best_bid < fair:
                orders.append(Order(product, best_bid, limit - pos))

        if best_ask is not None and limit + pos > 0:
            sell_price = best_ask - 1
            # Don't sell below fair (negative edge)
            if sell_price > fair:
                orders.append(Order(product, sell_price, -(limit + pos)))
            elif best_ask > fair:
                orders.append(Order(product, best_ask, -(limit + pos)))

        return orders

    def trade_pepper(self, state: TradingState) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        limit = 80
        od = state.order_depths.get(product)
        if not od:
            return []

        orders = []
        pos = state.position.get(product, 0)

        # Drift is +0.1/tick = +1000/day, extremely consistent
        # Optimal strategy: buy everything, never sell, ride the drift
        # At max position (80), we earn 80 * 0.1 = 8 per tick = 80K per day

        # Take ALL available asks — every buy is profitable due to drift
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                qty = min(-od.sell_orders[price], limit - pos)
                if qty > 0:
                    orders.append(Order(product, price, qty))
                    pos += qty

        # Post aggressive buy at best ask to fill remaining capacity ASAP
        # Paying the spread is trivial vs drift earnings
        if limit - pos > 0:
            if od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                orders.append(Order(product, best_ask, limit - pos))
            elif od.buy_orders:
                best_bid = max(od.buy_orders.keys())
                # Post above best bid to attract fills
                orders.append(Order(product, best_bid + 1, limit - pos))

        return orders

    def run(self, state: TradingState):
        result = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self.trade_osmium(state)
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result["INTARIAN_PEPPER_ROOT"] = self.trade_pepper(state)

        # Handle any other products that might appear (future rounds)
        for product in state.order_depths:
            if product not in result:
                result[product] = []

        conversions = 0
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data