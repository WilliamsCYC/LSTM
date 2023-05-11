import backtrader as bt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MyData(bt.feeds.PandasData):
    """
    in backtrader, it is necessary to define a class if user want to feed
    data with pre-defined columns rather than the necessary seven columns
    """
    '''put the names of pre-defined columns in a tuple'''
    lines = ('probability', )             # the last comma is necessary
    params_list = [
        # necessary seven columns
        ('datetime', 'trade_date'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'vol'),
        ('openinterest', 'openinterest')
    ]
    '''add pre-defined columns'''
    for i in lines:
        params_list.append((i, i))
    params = tuple(params_list)


class TestStrategy(bt.Strategy):
    """customize the strategy"""
    def __init__(self):
        self.signal = self.data.probability > 0.5
        self.order = None

    def next(self):
        """buy stock if not holding any position and my ML model predicts that the probability of a rise is over 0.5"""
        if not self.position:
            if self.signal:
                self.order = self.buy()
        # sell stock if holding position and my ML model predicts that the probability of a drop is over 0.5
        else:
            if not self.signal:
                self.order = self.close()


def backtest_results(df):
    """initialize engine"""
    cerebro = bt.Cerebro()
    data = MyData(dataname=df)
    '''feed data'''
    cerebro.adddata(data)
    '''add strategy'''
    cerebro.addstrategy(TestStrategy)
    '''each order uses 80% cash'''
    cerebro.addsizer(bt.sizers.PercentSizer, percents=80)
    '''initial cash'''
    cerebro.broker.setcash(100000.0)
    '''broker commission'''
    cerebro.broker.setcommission(commission=0.0003)
    '''slippage'''
    cerebro.broker.set_slippage_perc(perc=0.0001)
    '''record daily return and transaction points'''
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='daily_return')
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='trade')
    '''start engine'''
    result = cerebro.run(stdstats=False)
    result = result[0]
    '''get daily return and transaction points'''
    daily_return = pd.Series(result.analyzers.daily_return.get_analysis())
    transactions = result.analyzers.trade.get_analysis()
    return daily_return, transactions


def plot_performance(daily_return):

    """calculate metrics to plot table"""
    cumulative_return = (daily_return + 1).cumprod()
    max_return = cumulative_return.cummax()
    drawdown = - (cumulative_return - max_return) / max_return
    max_drawdown = drawdown.cummax()
    '''transform into dataframe'''
    cumulative_return_df = cumulative_return.to_frame(name='cumulative_return')
    max_drawdown_df = max_drawdown.to_frame(name='max_drawdown')
    risk_free_rate = 0.02
    sharpe_ratio = (daily_return.mean() * 252 - risk_free_rate) / (daily_return.std() * np.sqrt(252) + 1e-4)
    annual_return = cumulative_return[-1] ** (252/len(daily_return)) - 1
    calmar = (daily_return.mean() * 252 - risk_free_rate) / max_drawdown.max()

    cols_names = ['Sharpe ratio',
                  'Annual return',
                  'Cumulative returns',
                  'Annual volatility',
                  'Calmar ratio',
                  'Max drawdown',
                  ]

    cell_values = [sharpe_ratio.round(4),
                   annual_return.round(4),
                   cumulative_return[-1].round(4),
                   (daily_return.std() * np.sqrt(252)).round(4),
                   calmar.round(4),
                   max_drawdown.max().round(4)
                   ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]]
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=cols_names,
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=cell_values,
                align="left")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_return_df.index,
            y=cumulative_return_df['cumulative_return'],
            mode="lines",
            name='cumulative_return'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=max_drawdown_df.index,
            y=max_drawdown_df['max_drawdown'],
            mode="lines",
            name='max_drawdown'
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=600,
        title_text='Performance',
    )
    fig.update_xaxes(showgrid=False)
    fig.show()


def transactions_info(transactions):
    """trades is an OrderedDict, key is date, value is a list of ['amount', 'price', 'sid', 'symbol', 'value']"""
    buy_date = []
    buy_price = []
    sell_date = []
    sell_price = []
    for i, j in transactions.items():
        if j[0][0] >= 0:
            buy_date.append(i)
            buy_price.append(j[0][1])
        else:
            sell_date.append(i)
            sell_price.append(j[0][1])
    return buy_date, buy_price, sell_date, sell_price


def plot_transactions(df, transactions):
    buy_date, buy_price, sell_date, sell_price = transactions_info(transactions)

    fig = go.Figure(data=[go.Candlestick(x=df['trade_date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    showlegend=False)]
                    )

    fig.add_scatter(
        mode='markers',
        x=buy_date,
        y=buy_price,
        marker=dict(color='rgba(0,134,254,1)', size=10, symbol='arrow-up'),
        name='buy order'
    )

    fig.add_scatter(
        mode='markers',
        x=sell_date,
        y=sell_price,
        marker=dict(color='black', size=10, symbol='arrow-down'),
        name='sell order'
    )

    fig.update_xaxes(showgrid=False)

    fig.update_layout(
        height=600,
        title_text='Transactions',
    )

    fig.show()
