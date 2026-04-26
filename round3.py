import json
import math
from datamodel import OrderDepth, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import Any, List, Dict, Tuple

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

# ============================================================
# BLACK-SCHOLES
# ============================================================
def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))

def bs_call(S, K, T, sigma):
    if T <= 1e-8 or sigma <= 1e-8:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_delta(S, K, T, sigma):
    if T <= 1e-8 or sigma <= 1e-8:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def implied_vol(price, S, K, T):
    """Newton's method for IV"""
    if T <= 1e-8 or price <= max(0, S - K) + 0.01:
        return None
    sigma = 0.25  # initial guess
    for _ in range(50):
        p = bs_call(S, K, T, sigma)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
        vega = S * math.sqrt(T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        if vega < 1e-10:
            break
        sigma = sigma - (p - price) / vega
        if sigma <= 0.001:
            sigma = 0.001
    return sigma if 0.01 < sigma < 5.0 else None

VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

# ATM strikes used for TTE inference and smile fitting
ATM_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]


class Trader:

    def get_mid(self, od: OrderDepth):
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2, best_bid, best_ask
        return None, best_bid, best_ask

    def get_wall_mid(self, od: OrderDepth):
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is None and best_ask is None:
            return None, None, None
        if od.buy_orders:
            wall_bid = max(od.buy_orders.keys(), key=lambda p: od.buy_orders[p])
        else:
            wall_bid = best_bid
        if od.sell_orders:
            wall_ask = min(od.sell_orders.keys(), key=lambda p: -od.sell_orders[p])
        else:
            wall_ask = best_ask
        if wall_bid is not None and wall_ask is not None:
            fair = (wall_bid + wall_ask) / 2
        elif wall_bid is not None:
            fair = wall_bid + 8
        else:
            fair = wall_ask - 8
        return fair, best_bid, best_ask

    def infer_tte_and_smile(self, state: TradingState) -> Tuple[float, Dict[int, float]]:
        """Infer TTE from option prices and fit volatility smile.
        Returns (T_annual, {strike: fair_iv})"""
        
        # Get VE underlying price
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not ve_od:
            return 5.0 / 365.0, {}
        ve_mid, _, _ = self.get_mid(ve_od)
        if not ve_mid:
            return 5.0 / 365.0, {}
        S = ve_mid

        # Collect market mids for ATM options
        option_mids = {}
        for strike in ATM_STRIKES:
            prod = f"VEV_{strike}"
            od = state.order_depths.get(prod)
            if od:
                mid, _, _ = self.get_mid(od)
                if mid and mid > 0.5:
                    option_mids[strike] = mid

        if len(option_mids) < 3:
            return 5.0 / 365.0, {}

        # Find TTE that makes average ATM IV closest to 0.23
        best_T = 5.0 / 365.0
        best_err = 999
        for tte_days_x10 in range(30, 80):  # 3.0 to 8.0 days
            T = (tte_days_x10 / 10.0) / 365.0
            ivs = []
            for strike, market_mid in option_mids.items():
                iv = implied_vol(market_mid, S, strike, T)
                if iv:
                    ivs.append(iv)
            if len(ivs) >= 3:
                avg_iv = sum(ivs) / len(ivs)
                err = abs(avg_iv - 0.23)
                if err < best_err:
                    best_err = err
                    best_T = T

        # Now compute IV for all strikes at inferred TTE
        T = best_T
        strike_ivs = {}
        for strike in VOUCHER_STRIKES.values():
            prod = f"VEV_{strike}"
            od = state.order_depths.get(prod)
            if od:
                mid, _, _ = self.get_mid(od)
                if mid and mid > 0.5:
                    iv = implied_vol(mid, S, strike, T)
                    if iv:
                        strike_ivs[strike] = iv

        # Fit simple smile: use average of ATM IVs as baseline
        # For strikes without good IV, use 0.23
        for strike in VOUCHER_STRIKES.values():
            if strike not in strike_ivs:
                strike_ivs[strike] = 0.23

        return T, strike_ivs

    def trade_hydrogel(self, state: TradingState) -> List[Order]:
        product = "HYDROGEL_PACK"
        limit = 200
        od = state.order_depths.get(product)
        if not od:
            return []
        fair, best_bid, best_ask = self.get_wall_mid(od)
        if fair is None:
            return []
        orders = []
        pos = state.position.get(product, 0)

        # Take level 1 only (smart fill approach)
        if od.sell_orders:
            best_ask_price = min(od.sell_orders.keys())
            if best_ask_price < fair:
                qty = min(-od.sell_orders[best_ask_price], limit - pos)
                if qty > 0:
                    orders.append(Order(product, best_ask_price, qty))
                    pos += qty

        if od.buy_orders:
            best_bid_price = max(od.buy_orders.keys())
            if best_bid_price > fair:
                qty = min(od.buy_orders[best_bid_price], limit + pos)
                if qty > 0:
                    orders.append(Order(product, best_bid_price, -qty))
                    pos -= qty

        # Flatten at fair
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

        # Passive quotes
        if best_bid is not None and limit - pos > 0:
            buy_price = best_bid + 1
            if buy_price < fair:
                orders.append(Order(product, buy_price, limit - pos))
            elif best_bid < fair:
                orders.append(Order(product, best_bid, limit - pos))
        if best_ask is not None and limit + pos > 0:
            sell_price = best_ask - 1
            if sell_price > fair:
                orders.append(Order(product, sell_price, -(limit + pos)))
            elif best_ask > fair:
                orders.append(Order(product, best_ask, -(limit + pos)))
        return orders

    def trade_velvetfruit(self, state: TradingState) -> List[Order]:
        product = "VELVETFRUIT_EXTRACT"
        limit = 200
        od = state.order_depths.get(product)
        if not od:
            return []
        mid, best_bid, best_ask = self.get_mid(od)
        if mid is None:
            return []
        orders = []
        pos = state.position.get(product, 0)

        # Take below/above mid
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < mid:
                    qty = min(-od.sell_orders[price], limit - pos)
                    if qty > 0:
                        orders.append(Order(product, price, qty))
                        pos += qty
        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > mid:
                    qty = min(od.buy_orders[price], limit + pos)
                    if qty > 0:
                        orders.append(Order(product, price, -qty))
                        pos -= qty

        # Passive at best±1
        if best_bid is not None and limit - pos > 0:
            buy_price = best_bid + 1
            if buy_price < mid:
                orders.append(Order(product, buy_price, limit - pos))
        if best_ask is not None and limit + pos > 0:
            sell_price = best_ask - 1
            if sell_price > mid:
                orders.append(Order(product, sell_price, -(limit + pos)))
        return orders

    def trade_voucher(self, state: TradingState, product: str, T: float, strike_ivs: Dict[int, float]) -> List[Order]:
        limit = 300
        od = state.order_depths.get(product)
        if not od or product not in VOUCHER_STRIKES:
            return []

        strike = VOUCHER_STRIKES[product]
        sigma = strike_ivs.get(strike, 0.23)

        # Get VE underlying
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not ve_od:
            return []
        ve_mid, _, _ = self.get_mid(ve_od)
        if not ve_mid:
            return []
        S = ve_mid

        # BS fair price
        fair = bs_call(S, strike, T, sigma)
        if fair < 0.5:
            return []

        orders = []
        pos = state.position.get(product, 0)
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # Take mispriced (buy below fair, sell above fair)
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < fair - 1:
                    qty = min(-od.sell_orders[price], limit - pos)
                    if qty > 0:
                        orders.append(Order(product, price, qty))
                        pos += qty

        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > fair + 1:
                    qty = min(od.buy_orders[price], limit + pos)
                    if qty > 0:
                        orders.append(Order(product, price, -qty))
                        pos -= qty

        # Passive quotes around fair
        fair_int = round(fair)
        if best_bid is not None and limit - pos > 0:
            buy_price = min(best_bid + 1, fair_int - 1)
            if buy_price > 0:
                orders.append(Order(product, buy_price, limit - pos))

        if best_ask is not None and limit + pos > 0:
            sell_price = max(best_ask - 1, fair_int + 1)
            orders.append(Order(product, sell_price, -(limit + pos)))

        return orders

    def run(self, state: TradingState):
        result = {}

        # Infer TTE and volatility smile from current option prices
        T, strike_ivs = self.infer_tte_and_smile(state)

        # Delta-1 products
        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self.trade_hydrogel(state)
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            result["VELVETFRUIT_EXTRACT"] = self.trade_velvetfruit(state)

        # Vouchers
        for product in VOUCHER_STRIKES:
            if product in state.order_depths:
                result[product] = self.trade_voucher(state, product, T, strike_ivs)

        for product in state.order_depths:
            if product not in result:
                result[product] = []

        conversions = 0
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data