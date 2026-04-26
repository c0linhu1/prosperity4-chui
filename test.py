import json
import math
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


LIMITS = {
    "HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300, "VEV_5100": 300,
    "VEV_5200": 300, "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}

OPTION_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
OPTION_PRODUCTS = {f"VEV_{k}" for k in OPTION_STRIKES}
# ATM strikes where we actively scalp - these have strong mean reversion signals
ATM_STRIKES = {5000, 5100, 5200, 5300, 5400, 5500}
WALL_MID_PRODUCTS = {"HYDROGEL_PACK"}

FIXED_IV = 0.23
DAYS_PER_YEAR = 365

# Scalping params - tuned from data analysis:
# AC(theo_diff changes) ~ -0.48, meaning ~48% of each move reverts
# Spread cost on spread=2 strikes: 1 tick to enter + 1 tick to exit = 2 total
# Average |change| on signals > 0.3: ~0.6-0.9 depending on strike
# Expected profit per roundtrip: 0.48 * 0.7 - spread_cost ≈ needs careful sizing

SCALP_SIZE = 50          # contracts per scalp trade
SCALP_TAKE_SIZE = 20     # size when aggressively taking
INVENTORY_LIMIT = 100    # max abs position per option before we stop adding

from statistics import NormalDist
_N = NormalDist()


def bs_call(S, K, T, sigma):
    if T <= 1e-6 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N.cdf(d1) - K * math.exp(0) * _N.cdf(d2)

def bs_delta(S, K, T, sigma):
    if T <= 1e-6 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return _N.cdf(d1)


class Trader:

    def get_mid(self, od):
        bb = max(od.buy_orders.keys()) if od.buy_orders else None
        ba = min(od.sell_orders.keys()) if od.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2, bb, ba
        return None, bb, ba

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

    def market_make_delta1(self, state, product, td):
        limit = LIMITS.get(product, 200)
        od = state.order_depths.get(product)
        if not od:
            return []

        if product in WALL_MID_PRODUCTS:
            fair, bb, ba = self.get_wall_mid(od)
        else:
            fair, bb, ba = self.get_mid(od)
        if fair is None:
            return []

        orders = []
        pos = state.position.get(product, 0)

        # Phase 1: Take below/above fair
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

        # Phase 2: Flatten at fair
        fi = round(fair)
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

        # Phase 3: Passive quotes with inventory skew
        if bb is not None and ba is not None:
            spread = ba - bb
            inventory_ratio = pos / limit if limit > 0 else 0
            skew = round(inventory_ratio * 2)

            if spread >= 2:
                buy_price = bb + 1
                sell_price = ba - 1
                # Apply inventory skew
                buy_price = min(buy_price - min(skew, 0), fi)
                sell_price = max(sell_price - max(skew, 0), fi)
                if buy_price < sell_price:
                    if limit - pos > 0:
                        orders.append(Order(product, int(buy_price), limit - pos))
                    if limit + pos > 0:
                        orders.append(Order(product, int(sell_price), -(limit + pos)))
                else:
                    if pos > 0 and limit + pos > 0:
                        orders.append(Order(product, int(sell_price), -(limit + pos)))
                    elif pos < 0 and limit - pos > 0:
                        orders.append(Order(product, int(buy_price), limit - pos))
        elif bb is not None and limit - pos > 0:
            orders.append(Order(product, bb + 1, limit - pos))
        elif ba is not None and limit + pos > 0:
            orders.append(Order(product, ba - 1, -(limit + pos)))

        return orders


    def trade_options(self, state, td):
        """
        IV Scalping based on theo_diff CHANGES.
        
        Signal: theo_diff = option_mid - BS_fair(IV=0.23)
        The CHANGE in theo_diff from tick to tick has AC ≈ -0.48
        meaning ~48% of each move reverts next tick.
        
        Strategy:
        - If theo_diff went UP since last tick -> option temporarily overpriced -> SELL
        - If theo_diff went DOWN since last tick -> option temporarily underpriced -> BUY
        - Use spread >= 2 filter to avoid accidental taking
        
        On top of scalping: passive MM on ALL options with spread >= 2
        """
        # Get VE mid for BS calculations
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not ve_od:
            return {}
        ve_mid, ve_bb, ve_ba = self.get_mid(ve_od)
        if ve_mid is None:
            return {}

        # TTE calculation
        base_tte_days = td.get("btte", 5)
        intraday_frac = (state.timestamp / 100) / 10000
        tte = max((base_tte_days - intraday_frac) / DAYS_PER_YEAR, 1e-6)

        result = {}

        for strike in OPTION_STRIKES:
            product = f"VEV_{strike}"
            od = state.order_depths.get(product)
            if not od:
                continue

            limit = LIMITS[product]
            pos = state.position.get(product, 0)
            mid, bb, ba = self.get_mid(od)

            if bb is None or ba is None or mid is None:
                continue

            spread = ba - bb
            orders = []

            # ── Compute BS fair and theo_diff ──
            bs_fair = bs_call(ve_mid, strike, tte, FIXED_IV)
            theo_diff = mid - bs_fair

            # ── Load previous theo_diff ──
            td_key = f"d_{strike}"
            prev_diff = td.get(td_key)
            td[td_key] = theo_diff  # store for next tick

            is_atm = strike in ATM_STRIKES

            scalped = False
            if is_atm and prev_diff is not None and spread >= 2:
                change = theo_diff - prev_diff

                # After an UP change: option overpriced vs fair -> SELL at bid
                # After a DOWN change: option underpriced vs fair -> BUY at ask
                #
                # We TAKE (cross the spread) when the signal is strong enough
                # to overcome the spread cost. With AC=-0.48 and avg|change|~0.7,
                # expected reversion ~ 0.48 * 0.7 = 0.34 per unit.
                # On spread=2: cost to roundtrip = 2 ticks for size=1
                # So we need to trade passively where possible.

                # AGGRESSIVE TAKING: when change is large enough
                if change >= 1.0 and pos > -INVENTORY_LIMIT:
                    # Strong overpricing signal -> hit the bid aggressively
                    take_size = min(SCALP_TAKE_SIZE, INVENTORY_LIMIT + pos)
                    if take_size > 0:
                        orders.append(Order(product, bb, -take_size))
                        pos -= take_size
                        scalped = True

                elif change <= -1.0 and pos < INVENTORY_LIMIT:
                    # Strong underpricing signal -> lift the ask aggressively
                    take_size = min(SCALP_TAKE_SIZE, INVENTORY_LIMIT - pos)
                    if take_size > 0:
                        orders.append(Order(product, ba, take_size))
                        pos += take_size
                        scalped = True

                # PASSIVE SCALPING: post on the side the signal suggests
                # When change > 0 (overpriced): post sell at ask-1, and buy at bid (to catch reversion)
                # When change < 0 (underpriced): post buy at bid+1, and sell at ask (to catch reversion)
                if not scalped and abs(change) > 0.2:
                    if change > 0.2 and pos > -INVENTORY_LIMIT:
                        # Overpriced -> want to sell. Post sell at ask-1 (penny the ask)
                        sell_qty = min(SCALP_SIZE, INVENTORY_LIMIT + pos)
                        if sell_qty > 0 and spread >= 2:
                            orders.append(Order(product, ba - 1, -sell_qty))
                            scalped = True
                    elif change < -0.2 and pos < INVENTORY_LIMIT:
                        # Underpriced -> want to buy. Post buy at bid+1 (penny the bid)
                        buy_qty = min(SCALP_SIZE, INVENTORY_LIMIT - pos)
                        if buy_qty > 0 and spread >= 2:
                            orders.append(Order(product, bb + 1, buy_qty))
                            scalped = True


            if not scalped and spread >= 2:
                max_buy = min(limit - pos, SCALP_SIZE)
                max_sell = min(limit + pos, SCALP_SIZE)

                if spread >= 4:
                    # Wide spread: use BS fair as reference
                    bp = bb + 1
                    sp = ba - 1
                    if bp <= round(bs_fair) and max_buy > 0:
                        orders.append(Order(product, bp, max_buy))
                    if sp >= round(bs_fair) and max_sell > 0:
                        orders.append(Order(product, sp, -max_sell))
                else:
                    # Spread 2-3: simple penny
                    if max_buy > 0:
                        orders.append(Order(product, bb + 1, max_buy))
                    if max_sell > 0:
                        orders.append(Order(product, ba - 1, -max_sell))

            # ── INVENTORY FLATTENING ──
            # If we've accumulated too much, flatten toward zero
            if not scalped and abs(pos) > INVENTORY_LIMIT and spread >= 2:
                if pos > INVENTORY_LIMIT:
                    flatten_qty = min(pos - INVENTORY_LIMIT // 2, limit + pos)
                    if flatten_qty > 0:
                        orders.append(Order(product, bb, -flatten_qty))
                elif pos < -INVENTORY_LIMIT:
                    flatten_qty = min(-pos - INVENTORY_LIMIT // 2, limit - pos)
                    if flatten_qty > 0:
                        orders.append(Order(product, ba, flatten_qty))

            if orders:
                result[product] = orders

        return result

    def run(self, state: TradingState):
        result = {}

        # Load trader data
        td = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except:
                td = {}

        # ── Delta-1 products ──
        for product in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]:
            if product in state.order_depths:
                result[product] = self.market_make_delta1(state, product, td)

        # ── Options: IV scalping + passive MM ──
        option_orders = self.trade_options(state, td)
        result.update(option_orders)

        # Serialize trader data
        conversions = 0
        try:
            trader_data = json.dumps(td)
        except:
            trader_data = ""

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data