from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = ""

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []
            position = state.position.get(product, 0)
            limit = 80

            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid = (best_bid + best_ask) / 2

                if state.timestamp % 10000 == 0:
                    print(f"t={state.timestamp} {product} | bid={best_bid} ask={best_ask} mid={mid:.1f} pos={position}")

                # Take: buy anything below mid, sell anything above mid
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    ask_vol = order_depth.sell_orders[ask_price]  # negative
                    if ask_price < mid:
                        buy_qty = min(-ask_vol, limit - position)
                        if buy_qty > 0:
                            orders.append(Order(product, ask_price, buy_qty))
                            position += buy_qty

                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    bid_vol = order_depth.buy_orders[bid_price]  # positive
                    if bid_price > mid:
                        sell_qty = min(bid_vol, limit + position)
                        if sell_qty > 0:
                            orders.append(Order(product, bid_price, -sell_qty))
                            position -= sell_qty

                # Make: place passive orders inside the spread
                spread = best_ask - best_bid
                if spread > 2:
                    if position < limit:
                        orders.append(Order(product, best_bid + 1, min(10, limit - position)))
                    if position > -limit:
                        orders.append(Order(product, best_ask - 1, -min(10, limit + position)))

            result[product] = orders

        return result, conversions, trader_data