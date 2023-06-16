from backtesting import Strategy
from backtesting.lib import resample_apply
import pandas_ta as ta
import pandas as pd
import numpy as np
           
# ===================== START OF UNIVERSAL STRATEGY CLASS =====================

# ==================== UnderwaterStrategy ====================
class UnderwaterStrategy(Strategy):
    # All of the following variables can be used during optimization
    initial_position_size = 0.3
    percent_invested_threshold = 0.3
    atr_length = 14  # 14 days
    atr_multiplier = 0.5
    add_size = 0.1
    delay_period = 1000
    delta_time = 1000
    upper_bound_profit_target = 0.05
    lower_bound_loss_threshold = 0.02
    take_profit_loss_reduction = -0.1  # amount take profit is reduced by if the position is highly leveraged and we wish to trim
    deleverage_pct = 0.30  # This is the amount that the position is reduced by if the position is highly leveraged and we wish to trim
    bounce_multiplier = 1.5  # multiply the ATR Treshold by this amount to get the bounce threshold we are looking to bounce off a low "max pain" point before we reduce our position
    max_loss_threshold = -0.05  # if the position is down this much, and we are fully invested we will reduce our positio
    max_hold_length = None  # 720  # If not None, then number of minutes to hold a position before closing it. Enter None if you want to exit naturally
    trade_type: str = 'LONG_SHORT' # 'LONG_ONLY', 'SHORT_ONLY', 'LONG_SHORT'
       
    # Rest of your code goes here...
    def SIGNAL(self):
        return self.data.signal

    def ATR(self, df, length):
        return df.ta.atr(length=length)

    def bars_since_first_trade(self, use_for_indexing=False):
        """
        Calculate the number of bars since the first trade was entered.
        If use for indexing is true then it will return 1 if there are no trades.
        This way you can retrieve the last element of a list by applying a - to the return value.
        eg. self.equity[-self.bars_since_first_trade(use_for_indexing=True):] gives you the array of your account's equity since you entered the first trade that is currently open
        """
        if len(self.trades) > 0:
            self.first_trade_entry_bar = self.trades[0].entry_bar
            bars_since_first_trade = len(self.data.Close) - self.first_trade_entry_bar
            return bars_since_first_trade
        elif use_for_indexing:
            return 1
        else:
            return 0

    def custom_decay_func(
        self,
        x,
        delay_period,
        upper_bound_profit_target,
        lower_bound_loss_threshold,
        delta_time,
    ):
        """
        This function is used to calculate the decayed take profit x represents the number of bars since the first trade was entered
        x is the number of bars since the first trade was entered
        delay_period is the number of bars to wait before starting to decay
        upper_bound_profit_target is the upper bound of the take profit
        lower_bound_loss_threshold is the lower bound of the take profit
        delta_time is the number of bars over which to transition from the upper bound to the lower bound
        """
        if x <= delay_period:
            return upper_bound_profit_target
        elif delay_period < x < delay_period + delta_time:
            # Calculate the x value for the cos function
            transition_x = (x - delay_period) / delta_time * np.pi
            # Calculate the decayed take profit
            return (-np.cos(transition_x) + 1) / 2 * (
                lower_bound_loss_threshold - upper_bound_profit_target
            ) + upper_bound_profit_target
        else:
            return lower_bound_loss_threshold

    def init(self):

        super().init()
        self.signal = self.I(self.SIGNAL)
        self.atr = self.I(self.ATR, self.data.df, self.atr_length)
        self.daily_atr = resample_apply("1D", self.ATR, self.data.df, length=14)
        count_NaN = len(self.atr) - len(
            pd.Series(self.atr).dropna()
        )  # This is used for the progress bar
        self.length_of_data = (
            len(self.data.Close) - count_NaN
        )  # This is used for the progress bar
        self.equity_during_trade = []  # Keeps a list for the equity during the trade
        self.long_short_flag = (
            None  # This is used to keep track of whether we are long or short
        )
        self.price_at_last_trim = (
            0  # This is used to keep track of the price at the last trim
        )

    def next(self):
        super().next()

        price = self.data.Close[-1]
        position_value = self.position.size * price
        percent_invested = (
            position_value / self.equity
        )  # this will come in handy if we decide to change behavior once XX% is invested
        atr_threshold_pct = (
            self.atr_multiplier * self.daily_atr[-1] / price
        )  # This is the ATR threshold times a multiplier calculated as a percentage of price
        bars_since_trade_open = self.bars_since_first_trade(use_for_indexing=True)
        highly_leveraged = abs(percent_invested) > self.percent_invested_threshold
        # Calculate the decayed take profit
        decayed_take_profit = self.custom_decay_func(
            bars_since_trade_open,
            self.delay_period,
            self.upper_bound_profit_target,
            self.lower_bound_loss_threshold,
            self.delta_time,
        )

        # Calculate the lowest the equity ever got since we have been in a trade
        if self.position:  
            self.lowest_equity_during_trade = min(self.lowest_equity_during_trade, self.equity)
        else:
            self.lowest_equity_during_trade = float('inf')  # Reset to a very high number when not in a position

        low_point_in_trade = min(self.equity_during_trade, default=0)

        # Calculate the percentage bounce from the low point in the trade this is based on equity so works for long or short
        bounce_from_low_pct = (lambda x: (self.equity - x) / x if x != 0 else 0)(
            low_point_in_trade
        )  # The lambda function is used to avoid a divide by zero error

        # Opening a Trade on a signal from the LSTM
        if not self.position:
            if self.signal == 1 and self.trade_type in ['LONG_ONLY', 'LONG_SHORT']:
                self.buy(size=self.initial_position_size)
                self.long_short_flag = 1
            elif self.signal == -1 and self.trade_type in ['SHORT_ONLY', 'LONG_SHORT']:
                self.sell(size=self.initial_position_size)
                self.long_short_flag = -1

        
        # Closing a trade if max hold time has been reached
        if self.position and self.max_hold_length is not None and bars_since_trade_open > self.max_hold_length:
            self.position.close()
        # If we are in a short trade and the account is fully invested and the loss is greater than the max loss threshold then close the trade
        if self.long_short_flag == -1 and (abs(percent_invested) > 1 and self.position.pl_pct < self.max_loss_threshold):
            self.position.close() # If you want a partial close add `portion=self.deleverage_pct`
        
        # If we are in a trade and it meets our profit criteria then close the trade
        if self.position.pl_pct > decayed_take_profit:
            self.position.close()
            self.price_at_last_trim = 0
            # print(f'Closing at {price} at {self.data.index[-1]} Position PNL is {self.position.pl}')

        # If we are in a trade and it meets our loss criteria then close a portion of the trade on a bounce from the low point
        if self.position and self.position.pl_pct < -atr_threshold_pct:
            # Check to see if we are also down on our last trade
            if self.trades[-1].pl_pct < -atr_threshold_pct:
                if self.long_short_flag == 1:
                    self.buy(size=self.add_size)
                elif self.long_short_flag == -1:
                    self.sell(size=self.add_size)
        
        # If we are totally upside down then reduce the position size
        if self.position and self.position.pl_pct < decayed_take_profit + self.take_profit_loss_reduction:
            if self.trades[-1].pl_pct < -atr_threshold_pct: # if we are down more than 1 ATR threshold then reduce the position size and take a tiny loss
                self.position.close(portion=self.deleverage_pct)
                # print(f'Closing at {price} at {self.data.index[-1]} Position PNL is {self.position.pl}')
        
        # Deleverage if we are over a certain percent invested and the market is recovering from the low point
        if (
            highly_leveraged
            and bounce_from_low_pct > self.bounce_multiplier * atr_threshold_pct
        ):

            if (self.position.pl_pct > decayed_take_profit + self.take_profit_loss_reduction):
                if (self.long_short_flag == 1) and (self.price_at_last_trim == 0 or price > self.price_at_last_trim *(1 + 2 * atr_threshold_pct)):
                    self.position.close(portion=self.deleverage_pct)
                    # Keep track of the price to avoid trimming too often
                    self.price_at_last_trim = price 
                    # print(f'Deleveraging at {price} at {self.data.index[-1]} Position PNL is {self.position.pl}')
                elif (self.long_short_flag) == -1 and (self.price_at_last_trim == 0 or price < self.price_at_last_trim *(1 - 0.5 * atr_threshold_pct)):
                    self.position.close(portion=self.deleverage_pct)
                    # Keep track of the price to avoid trimming too often
                    self.price_at_last_trim = price 
                    # print(f'Deleveraging at {price} at {self.data.index[-1]} Position PNL is {self.position.pl}')