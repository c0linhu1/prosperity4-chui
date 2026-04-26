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
BASE_TAKE_LIMIT = 60   # base inventory cap for takes
FLATTEN_PCT = 0.95     # flatten last 5% (conservative, not overfit)


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
        td = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except:
                td = {}

        product = "HYDROGEL_PACK"
        od = state.order_depths.get(product)

        if od:
            orders = []
            pos = state.position.get(product, 0)
            fair, bb, ba = self.get_wall_mid(od)

            if fair is not None:
                fi = round(fair)
                ts = state.timestamp

                # Update from own_trades
                realized = td.get("r", 0.0)
                prev_fills = set(tuple(x) for x in td.get("f", []))
                if state.own_trades.get(product):
                    for t in state.own_trades[product]:
                        key = (t.price, t.quantity, t.timestamp)
                        if key not in prev_fills:
                            prev_fills.add(key)
                            if t.buyer == "SUBMISSION":
                                realized -= t.price * t.quantity
                            else:
                                realized += t.price * t.quantity
                td["r"] = realized
                td["f"] = [list(x) for x in prev_fills]

                # ── Compute MTM PnL ──
                mid = (bb + ba) / 2 if bb is not None and ba is not None else fair
                mtm_pnl = realized + pos * mid

                # ── Risk scaling: tighten inventory when PnL is high ──
                # The more PnL we've accumulated, the less risk we should take.
                # At 0 PnL: take_limit = 60 (normal)
                # At 2000 PnL: take_limit = 40
                # At 4000+ PnL: take_limit = 20
                # This way we protect gains without needing to know timing.
                if mtm_pnl > 4000:
                    take_limit = 20
                elif mtm_pnl > 2000:
                    take_limit = 40
                else:
                    take_limit = BASE_TAKE_LIMIT

                max_ts = td.get("mt", 0)
                if ts > max_ts:
                    max_ts = ts
                td["mt"] = max_ts
                day_end = 999900 if max_ts > 100000 else 99900

                flattening = ts >= day_end * FLATTEN_PCT

                if not flattening:
                    # Phase 1: Takes (capped at ±take_limit)
                    if od.sell_orders:
                        for price in sorted(od.sell_orders.keys()):
                            if price < fair and pos < take_limit:
                                qty = min(-od.sell_orders[price], LIMIT - pos, max(take_limit - pos, 0))
                                if qty > 0:
                                    orders.append(Order(product, price, qty))
                                    pos += qty

                    if od.buy_orders:
                        for price in sorted(od.buy_orders.keys(), reverse=True):
                            if price > fair and pos > -take_limit:
                                qty = min(od.buy_orders[price], LIMIT + pos, max(take_limit + pos, 0))
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

                    # Phase 3: Passive quotes (exact v7 — full LIMIT size)
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
                        if od.buy_orders:
                            for price in sorted(od.buy_orders.keys(), reverse=True):
                                if pos <= 0: break
                                qty = min(od.buy_orders[price], pos)
                                if qty > 0:
                                    orders.append(Order(product, price, -qty))
                                    pos -= qty
                        if pos > 0 and bb is not None:
                            orders.append(Order(product, bb + 1, -pos))

                    elif pos < 0:
                        if od.sell_orders:
                            for price in sorted(od.sell_orders.keys()):
                                if pos >= 0: break
                                qty = min(-od.sell_orders[price], -pos)
                                if qty > 0:
                                    orders.append(Order(product, price, qty))
                                    pos += qty
                        if pos < 0 and ba is not None:
                            orders.append(Order(product, ba - 1, -pos))

            result[product] = orders

        conversions = 0
        try:
            trader_data = json.dumps(td)
        except:
            trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data