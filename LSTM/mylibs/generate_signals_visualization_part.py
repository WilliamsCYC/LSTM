import numpy as np
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


'''----------MACD----------'''


def macd_golden_cross(df):
    """
    MACD golden cross: short term moving average cross above long term moving average
    and both moving average is lower than 0
    """
    golden_cross = np.where((df['macd_dif'] > df['macd_dea']) &
                            (df['macd_dif'].shift() < df['macd_dea'].shift()), 1, 0)

    '''if golden cross happens when dif and dea are close to zero, recognize this as a stronger signal'''
    golden_cross_enhanced = np.where((golden_cross == 1) &
                                     (15 < df['macd_dif']) & (df['macd_dif'] < 15) &
                                     (15 < df['macd_dea']) & (df['macd_dea'] < 15), 3, 0)

    '''some traders believe that dif line crossing over zero is a buy signal'''
    dif_cross_over_zero = np.where((df['macd_dif'] > 0) &
                                   (df['macd_dif'].shift() < 0), 1, 0)

    '''some traders believe that dea line crossing over zero is an even stronger buy signal'''
    dea_cross_over_zero = np.where((df['macd_dea'] > 0) &
                                   (df['macd_dea'].shift() < 0), 3, 0)

    return golden_cross + golden_cross_enhanced + dif_cross_over_zero + dea_cross_over_zero


def macd_death_cross(df):
    """
    MACD death cross: short term moving average cross below long term moving average
    and both moving average is greater than 0
    """
    death_cross = np.where((df['macd_dif'] < df['macd_dea']) &
                           (df['macd_dif'].shift() > df['macd_dea'].shift()) &
                           (df['macd_dif'] > 0) &
                           (df['macd_dea'] > 0), 1, 0)
    return death_cross


def macd_bullish_divergence(df):
    """
    A bullish divergence appears when MACD forms two rising lows that correspond with two falling lows on the price
    first, find the lows on price
    if one day's close price is lower than last and next 15 day's minimum close price, label that day as local minimum
    """
    local_min_index = np.zeros(len(df))
    for i in range(15, len(df) - 14):
        if df[['close']].iloc[i, 0] == df[['close']].iloc[i - 15:i + 16, 0].min():
            local_min_index[i] = 1

    ''' 
    get the close price and MACD of local minimum
    use last_local_min to contain the close price of the last local minimum and last_local_min_macd to 
    contain the macd of the last local minimum
    for example, if today is 2021-06-20 and last local minimum is on 2021-06-01 and the price is 10$, 
    then last_local_min['2021-06-20'] == 10
    '''
    last_local_min = np.zeros(len(df))
    last_local_min_macd = np.zeros(len(df))
    for i in range(15, len(df)):
        if local_min_index[i - 15] == 1:
            last_local_min[i] = df[['close']].iloc[i - 15, 0]
            last_local_min_macd[i] = df[['macd']].iloc[i - 15, 0]
    for i in range(1, len(last_local_min)):
        if last_local_min[i] == 0:
            last_local_min[i] = last_local_min[i - 1]
            last_local_min_macd[i] = last_local_min_macd[i - 1]

    divergence = np.zeros(len(df))
    for i in range(len(df)):
        '''the minimum price in the last 10 days is lower than last local minimum'''
        if df[['close']].iloc[i - 10:i, 0].min() < last_local_min[i]:
            '''that minimum price should also be the lowest in past 20 days'''
            if df[['close']].iloc[i - 10:i, 0].min() == df[['close']].iloc[i - 20:i, 0].min():
                '''that minimum MACD should also be the lowest in past 8 days'''
                if df[['macd']].iloc[i - 8:i, 0].min() == df[['macd']].iloc[i - 8:i, 0].min():
                    '''today's price is greater than the minimum price in the last 10 days'''
                    if df[['close']].iloc[i, 0] > df[['close']].iloc[i - 10:i, 0].min():
                        '''today's macd is higher than the minimum macd in the last 10 days'''
                        if df[['macd']].iloc[i, 0] > df[['macd']].iloc[i - 10:i, 0].min():
                            '''the MACD of minimum price in the last 10 days is greater than the MACD of
                            the last local minimum'''
                            if df[['macd']].iloc[i - 10:i, 0].min() > last_local_min_macd[i]:
                                divergence[i] = 1
    return divergence, local_min_index


def macd_bearish_divergence(df):
    """
    A bearish divergence appears when MACD forms two falling highs that correspond with two rising highs on the price
    first, find the highs on price
    if one day's close price is higher than last and next 15 day's maximum close price,
    label that day as local maximum
    """
    local_max_index = np.zeros(len(df))
    for i in range(15, len(df) - 14):
        if df[['close']].iloc[i, 0] == df[['close']].iloc[i - 15:i + 16, 0].max():
            local_max_index[i] = 1

    '''get the close price and MACD of local maximum'''
    last_local_max = np.zeros(len(df))
    last_local_max_macd = np.zeros(len(df))
    for i in range(15, len(df)):
        if local_max_index[i - 15] == 1:
            last_local_max[i] = df[['close']].iloc[i - 15, 0]
            last_local_max_macd[i] = df[['macd']].iloc[i - 15, 0]
    for i in range(1, len(last_local_max)):
        if last_local_max[i] == 0:
            last_local_max[i] = last_local_max[i - 1]
            last_local_max_macd[i] = last_local_max_macd[i - 1]

    divergence = np.zeros(len(df))
    for i in range(len(df)):
        '''the maximum price in the last 10 days is higher than last local maximum'''
        if df[['close']].iloc[i - 10:i, 0].max() > last_local_max[i]:
            '''that maximum price should also be the highest in past 20 days'''
            if df[['close']].iloc[i - 10:i, 0].max() == df[['close']].iloc[i - 20:i, 0].max():
                '''that maximum MACD should also be the highest in past 8 days'''
                if df[['macd']].iloc[i - 8:i, 0].max() == df[['macd']].iloc[i - 8:i, 0].max():
                    '''today's price is lower than the maximum price in the last 10 days'''
                    if df[['close']].iloc[i, 0] < df[['close']].iloc[i - 10:i, 0].max():
                        '''today's macd is lower than the maximum macd in the last 10 days'''
                        if df[['macd']].iloc[i, 0] < df[['macd']].iloc[i - 10:i, 0].max():
                            '''the MACD of maximum price in the last 10 days is lower than the MACD of
                            the last local maximum'''
                            if df[['macd']].iloc[i - 10:i, 0].max() < last_local_max_macd[i]:
                                divergence[i] = 1
    return divergence, local_max_index


'''----------Bollinger bands----------'''


def bollinger_bands(df):
    """when close price breaks up the upper band"""
    break_up = np.where(df['close'] > df['boll_upper'], 1, 0)
    """when close price breaks down the lower band"""
    break_down = np.where(df['close'] < df['boll_lower'], 1, 0)
    return break_up, break_down


'''----------Stochastic RSI----------'''


def stochrsi(df):
    current_rsi = ta.momentum.StochRSIIndicator(df.close, 14, 3, 3).stochrsi()
    '''when RSI is higher than 0.8'''
    stochrsi_overbought = np.where(current_rsi > 0.8, 1, 0)
    '''when RSI is lower than 0.2'''
    stochrsi_oversold = np.where(current_rsi < 0.2, 1, 0)
    return stochrsi_overbought, stochrsi_oversold, current_rsi


'''----------Others----------'''


def margin_long_and_short(df):
    """the amount people buy on margin is higher than the amount people repay their borrowing"""
    margin_long = np.where(df['net_margin_long'] >= 0, 1, 0)
    '''the volume people short is higher than the volume people repay their stock'''
    short = np.where(df['net_short'] >= 0, 1, 0)
    return margin_long, short


def net_big_order_signal(df):
    """the net volume of orders over 200,000 RMB is higher than its average value plus one standard deviation"""
    net_big_order = df['net_large_order'] + df['net_enormous_order']
    threshold = net_big_order.rolling(252).mean() + net_big_order.rolling(252).std()*1
    net_big_order_signal_ = np.where(net_big_order > threshold, 1, 0)
    return net_big_order_signal_


def high_volume(df):
    """trading volume is higher than its average value plus one standard deviation"""
    threshold = df['vol'].rolling(252).mean() + df['vol'].rolling(252).std()*1
    high_volume_ = np.where(df['vol'] > threshold, 1, 0)
    return high_volume_


def large_over_small(df):
    """the net volume of orders over 200,000 RMB is higher than the net volume of orders below 200,000 RMB"""
    aux = df['net_large_order'] + df['net_enormous_order'] - df['net_small_order'] - df['net_medium_order']
    large_over_small_ = np.where(aux > 0, 1, 0)
    return large_over_small_


'''----------integrate signals----------'''


def integrate(df):
    signals = pd.DataFrame(index=df.index)
    signals['macd_golden_cross'] = macd_golden_cross(df)
    signals['macd_death_cross'] = macd_death_cross(df)
    signals['macd_bullish_divergence'], signals['local_min_index'] = macd_bullish_divergence(df)
    signals['macd_bearish_divergence'], signals['local_max_index'] = macd_bearish_divergence(df)
    signals['margin_long'], signals['short'] = margin_long_and_short(df)
    signals['net_big_order_signal'] = net_big_order_signal(df)
    signals['high_volume'] = high_volume(df)
    signals['large_over_small'] = large_over_small(df)
    signals['boll_break_up'], signals['boll_break_down'] = bollinger_bands(df)
    signals['stochrsi_overbought'], signals['stochrsi_oversold'], signals['current_rsi'] = stochrsi(df)
    signals['close'] = df['close']
    return signals


'''----------Visualization----------'''


def plot_transactions(df, signals):
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"type": "Candlestick"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]]
    )
    '''Plot 1'''
    '''candle'''
    fig.add_trace(
        go.Candlestick(x=df.index,
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close'],
                       showlegend=False),
        row=1, col=1
    )
    '''Bollinger bands'''
    fig.add_trace(go.Scatter(x=df.index, y=df['boll_upper'], mode='lines', legendgroup='group1',
                             legendgrouptitle_text='Bollinger bands', name='bollinger upper band'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['boll_mid'], mode='lines', legendgroup='group1',
                             name='bollinger upper band'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['boll_lower'], mode='lines', legendgroup='group1',
                             name='bollinger lower band'), row=1, col=1)
    '''Bollinger break up'''
    x_aux = signals[signals['boll_break_up'] > 0].index
    y_aux = df[signals['boll_break_up'] > 0].close
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode='markers', legendgroup='group1', name='Bollinger break up',
                             marker=dict(color='purple', size=8, symbol='arrow-down')),
                  row=1, col=1)
    '''Bollinger break down'''
    x_aux = signals[signals['boll_break_down'] > 0].index
    y_aux = df[signals['boll_break_down'] > 0].close
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode='markers', legendgroup='group1', name='Bollinger break down',
                             marker=dict(color='black', size=8, symbol='arrow-up')),
                  row=1, col=1)
    '''serve for MACD'''
    '''local minimum'''
    x_aux = signals[signals['local_min_index'] > 0].index
    y_aux = df[signals['local_min_index'] > 0].close
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode='markers', legendgroup='group2',
                             legendgrouptitle_text='MACD',
                             name='local minimum', marker=dict(color='purple', size=8, symbol='arrow-up')),
                  row=1, col=1)
    '''local maximum'''
    x_aux = signals[signals['local_max_index'] > 0].index
    y_aux = df[signals['local_max_index'] > 0].close
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode='markers', legendgroup='group2', name='local maximum',
                             marker=dict(color='black', size=8, symbol='arrow-down')),
                  row=1, col=1)

    '''Plot 2'''
    '''MACD'''
    fig.add_trace(go.Bar(x=df.index, y=df['macd'], legendgroup='group2', name='MACD'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_dif'], mode="lines", legendgroup='group2', name='MACD dif'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_dea'], mode="lines", legendgroup='group2', name='MACD dea'),
                  row=2, col=1)
    '''MACD golden cross'''
    x_aux = signals[signals['macd_golden_cross'] > 0].index
    y_aux = df[signals['macd_golden_cross'] > 0].macd_dif
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode='markers', legendgroup='group2', name='MACD golden cross',
                             marker=dict(color='purple', size=8, symbol='x')),
                  row=2, col=1)
    '''MACD death cross'''
    x_aux = signals[signals['macd_death_cross'] > 0].index
    y_aux = df[signals['macd_death_cross'] > 0].macd_dif
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode='markers', legendgroup='group2', name='MACD death cross',
                             marker=dict(color='black', size=8, symbol='x')),
                  row=2, col=1)
    '''MACD bullish divergence'''
    x_aux = signals[signals['macd_bullish_divergence'] > 0].index
    y_aux = df[signals['macd_bullish_divergence'] > 0].macd_dif
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode='markers', legendgroup='group2', name='MACD bullish divergence',
                             marker=dict(color='green', size=8, symbol='arrow-up')),
                  row=2, col=1)
    '''MACD bearish divergence'''
    x_aux = signals[signals['macd_bearish_divergence'] > 0].index
    y_aux = df[signals['macd_bearish_divergence'] > 0].macd_dif
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode='markers', legendgroup='group2', name='MACD bearish divergence',
                             marker=dict(color='red', size=8, symbol='arrow-down')),
                  row=2, col=1)

    '''Plot 3'''
    '''RSI'''
    fig.add_trace(go.Scatter(x=signals.index, y=signals['current_rsi'], mode="lines", legendgroup='group3',
                             legendgrouptitle_text='RSI', name='RSI'),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=signals.index, y=[0.8 for i in range(len(signals))], mode="lines",
                             legendgroup='group3', name='80%', line=dict(dash='dash')),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=signals.index, y=[0.2 for i in range(len(signals))], mode="lines",
                             legendgroup='group3', name='20%', line=dict(dash='dash')),
                  row=3, col=1)

    fig.layout.yaxis2.update({'title': 'MACD'})
    fig.layout.yaxis3.update({'title': 'RSI'})

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.04),
            type="date")
    )

    fig.update_xaxes(showgrid=False)

    fig.update_layout(
        height=1000,
        title_text='Market data',
    )

    fig.show()


def plot_alternative(signals):
    fig = make_subplots(
                        rows=3, cols=1,
                        row_heights=[0.25, 0.25, 0.25],
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        specs=[[{"type": "scatter"}],
                               [{"type": "scatter"}],
                               [{"type": "scatter"}]]
    )

    x_aux = signals[signals['sentiment'] > 0].index
    y_aux = signals[signals['sentiment'] > 0]['sentiment']
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode="markers", name='News sentiment'),
                  row=1, col=1)
    x_aux = signals[signals['rating'] > 0].index
    y_aux = signals[signals['rating'] > 0]['rating']
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode="markers", name='Analyst rating'),
                  row=2, col=1)
    x_aux = signals[signals['dif_forecast_current'] > 0].index
    y_aux = signals[signals['dif_forecast_current'] > 0]['dif_forecast_current']
    fig.add_trace(go.Scatter(x=x_aux, y=y_aux, mode="markers", name='Forecast-current'),
                  row=3, col=1)

    fig.layout.yaxis1.update({'title': 'Sentiment'})
    fig.layout.yaxis2.update({'title': 'Rating'})
    fig.layout.yaxis3.update({'title': 'Forecast - Current'})

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.04),
            type="date")
    )

    fig.update_layout(
        height=1000,
        title_text='Alternative data',
    )
    fig.show()
