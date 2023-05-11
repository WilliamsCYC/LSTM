import numpy as np
import pandas as pd
import datetime
'''cemotion is a Chinese sentiment analysis package'''
from cemotion import Cemotion
import ta

# tushare is a famous package in China to download stock data
# https://tushare.pro
import tushare as ts
pro = ts.pro_api('c12abd8696482b6d604d47293faef7efe9fbfe14db6b18b1669eeab1')


'''----------download raw data----------'''


def get_tushare_data(ts_code, start_date='20140101', end_date='20211231'):
    """basic ohlc, volume and some technical indicators"""
    """qfq means adjusted for dividend or split, etc"""
    fields = 'trade_date,open_qfq,high_qfq,low_qfq,close_qfq,vol,amount,macd_dif,macd_dea,macd,kdj_k,\
              kdj_d,kdj_j,rsi_6,rsi_12,rsi_24,boll_upper,boll_mid,boll_lower,cci'
    df = pro.stk_factor(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
    df = df.rename(columns={'open_qfq': 'open', 'high_qfq': 'high', 'low_qfq': 'low', 'close_qfq': 'close'})

    fields_2 = 'trade_date,turnover_rate_f,volume_ratio,pe,pb,ps'
    df_2 = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields_2)

    '''
    trade amount and price for different size of orders
    buy_sm_vol/sell_sm_vol: order lower than 50,000 RMB
    buy_md_vol/sell_md_vol: order from 50,000 to 200,000 RMB
    buy_lg_vol/sell_lg_vol: order from 200,000 to 1 million RMB
    buy_elg_vol/sell_elg_vol: order over 1 million RMB
    '''
    fields_3 = 'trade_date,buy_sm_vol,sell_sm_vol,buy_md_vol,sell_md_vol,\
                buy_lg_vol,sell_lg_vol,buy_elg_vol,sell_elg_vol'
    df_3 = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields_3)

    '''
    north money and south money are hot topics in mainland China (people tall more about north money),
    north money is how much global investors are investing in mainland China stock market through Hong Kong
    and south money is inverse
    '''
    df_4 = pd.DataFrame()
    fields_4 = 'trade_date,north_money,south_money'
    '''these data API has request limitation, so divide the total amount'''
    for i in range(2014, 2022):
        start = str(i) + '0101'
        end = str(i) + '1231'
        df_aux = pro.moneyflow_hsgt(start_date=start, end_date=end, fields=fields_4)
        df_4 = pd.concat([df_4, df_aux])
    df_4 = df_4.fillna(method='ffill')

    '''short sell and buy on margin'''
    '''
    rzmre: amount of long orders on margin
    rzche: amount of margin repaid
    rqmcl: volume of short sell
    rqchl: volume of closing short position
    '''
    fields_5 = 'trade_date,rzmre,rzche,rqmcl,rqchl'
    df_5 = pro.margin_detail(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields_5)

    df = pd.merge(df, df_2, how='left', on='trade_date').sort_values(by='trade_date')
    df = pd.merge(df, df_3, how='left', on='trade_date').sort_values(by='trade_date')
    df = pd.merge(df, df_4, how='left', on='trade_date').sort_values(by='trade_date')
    df = pd.merge(df, df_5, how='left', on='trade_date').sort_values(by='trade_date')

    '''log return'''
    df['ret'] = np.log(df['close']) - np.log(df['close'].shift())
    return df


'''----------construct more features and process----------'''


def add_indicator(data):
    data['change'] = data.close - data.close.shift(1)
    data['mtm_10'] = data.close/data.close.shift(10) * 100
    data['roc_10'] = (data.close - data.close.shift(10))/data.close.shift(10) * 100

    wnr = ta.momentum.WilliamsRIndicator(data.high, data.low, data.close, 9)
    data['wnr_9'] = wnr.williams_r()

    slow = ta.momentum.StochRSIIndicator(data.close, 14, 3, 3)
    data['slowk'] = slow.stochrsi_k()
    data['slowd'] = slow.stochrsi_d()

    data['adosc'] = (data.high - data.close.shift())/(data.high - data.low + 1e-4)
    data['ar_26'] = (data.high.rolling(26).sum() - data.open.rolling(26).sum()) / \
                    (data.open.rolling(26).sum() - data.low.rolling(26).sum() + 1e-4)
    data['br_26'] = (data.high.rolling(26).sum() - data.open.shift().rolling(26).sum()) / \
                    (data.open.shift().rolling(26).sum() - data.low.rolling(26).sum() + 1e-4)
    data['bias_20'] = (data.close - data.close.rolling(20).mean()) / data.close.rolling(20).mean()
    return data


def add_yz(data, n=10):
    """calculate YZ estimator of n days"""
    C_0 = data['close'].shift()
    o = np.log(data['open']/C_0)
    u = np.log(data['high']/data['open'])
    d = np.log(data['low']/data['open'])
    c = np.log(data['close']/data['open'])
    # V_RS
    V_RS = (u*(u-c)+d*(d-c)).rolling(n).mean()
    # V_o
    V_o = (o - o.rolling(n).mean()).rolling(n).var()
    # V_c
    V_c = (c - c.rolling(n).mean()).rolling(n).var()
    # V_YZ
    k = 0.34/(1.34+(n+1)/(n-1))
    data['V_YZ'] = V_o + k*V_c + (1-k)*V_RS
    return data


def add_statistical(data):
    for i in [10, 30, 50, 100, 200]:
        data[f'Std_{str(i)}'] = data['ret'].rolling(i).std()
    return data


def add_candle(data):
    data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
    data['lower_shadow'] = data[['open', 'close']].max(axis=1) - data['low']
    data['body'] = abs(data['close'] - data['open'])
    data['whole_length'] = data['high'] - data['low']
    return data


def margin_and_short(data):
    """calculate the net amount/volume of long order on margin and short order"""
    '''net_margin_long = long amount with margin - repayment of margin of long position'''
    data['net_margin_long'] = data['rzmre'] - data['rzche']
    '''net_short = short volume - repayment of stock (volume) of short position'''
    data['net_short'] = data['rqmcl'] - data['rqchl']
    data.drop(['rzmre', 'rzche', 'rqmcl', 'rqchl'], axis=1, inplace=True)
    return data


def order_size(data):
    """calculate the net volume of certain order size"""
    data['net_small_order'] = data['buy_sm_vol'] - data['sell_sm_vol']
    data['net_medium_order'] = data['buy_md_vol'] - data['sell_md_vol']
    data['net_large_order'] = data['buy_lg_vol'] - data['sell_lg_vol']
    data['net_enormous_order'] = data['buy_elg_vol'] - data['sell_elg_vol']
    data.drop(['buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
               'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol'], axis=1, inplace=True)
    return data


def peer_stock(data, ts_code):
    """close price of peer stocks"""
    data = data.copy()
    peer_stock_list = ['600519.SH', '000858.SZ', '000596.SZ', '000568.SZ', '600779.SH', '600809.SH', '002304.SZ']
    peer_stock_list.remove(ts_code)
    for i in peer_stock_list:
        aux_df = pro.stk_factor(ts_code=i, start_date='20140101', end_date='20211231', fields='trade_date,close_qfq')
        aux_df = aux_df.sort_values(by='trade_date')
        aux_df = aux_df.rename(columns={'close_qfq': 'close'})
        aux_df = aux_df.rename(columns={'close': i})
        data = pd.merge(data, aux_df, on='trade_date', how='left')
        '''with few null values, use ffill'''
        data[i] = data[i].fillna(method='ffill')
    return data


def construct_and_process(data, ts_code):
    """integrate everything"""
    data = add_yz(data)
    data = add_candle(data)
    data = add_statistical(data)
    data = margin_and_short(data)
    data = order_size(data)
    data = peer_stock(data, ts_code)
    data = add_indicator(data)
    data['trade_date'] = pd.to_datetime(data['trade_date'])
    return data


'''----------For alternative data----------'''


def push_back_to_trading_day(data, df):
    """if the report is released on a non-trading day, push it back to the last trading day"""
    '''data should be a dataframe with trade_date in the first column'''
    if data.columns[0] != 'trade_date':
        '''I just randomly pick an error type'''
        raise TypeError

    for i in range(len(data)):
        date = data.iloc[i, 0]
        '''set j to avoid endless loop'''
        j = 0
        ''''date' is the date of some events happened'''
        '''verify whether the date is one of the trading day'''
        while date not in df.index.values:
            '''if not, push it back one day and verify it again, loop till the date is a trading day'''
            date = date - datetime.timedelta(days=1)
            '''set j to avoid endless loop'''
            j += 1
            if j == 100:
                break
        data.iloc[i, 0] = date
    return data


def get_rating(ts_code, df):
    ts_code_sp = ts_code[:6]
    """raw data are in two separate csv files"""
    rating = pd.read_csv(r'./data/forecast/rating1.csv', low_memory=False).drop([0, 1])
    rating_ = pd.read_csv(r'./data/forecast/rating2.csv', low_memory=False).drop([0, 1])
    rating = pd.concat([rating, rating_])
    '''select lines with respect to target stock'''
    rating = rating[rating['Stkcd'] == ts_code_sp].copy()
    ''''Rptdt' is the date of declaration date of that report'''
    ''''Stdrank' is analysts' recommendation'''
    rating = rating[['Rptdt', 'Stdrank']].copy()
    rating = rating.rename(columns={'Stdrank': 'rating', 'Rptdt': 'trade_date'})
    rating['trade_date'] = pd.to_datetime(rating['trade_date'])
    '''this data source only has three rating '买入'(buy), '增持'(increase position), '中性'(neutral)'''
    '''give value of 2 for '买入'(buy), 1 for '增持'(increase position) and 0 for '中性'(neutral)'''
    for i in range(len(rating)):
        if rating.iloc[i, 1] == '增持':
            rating.iloc[i, 1] = 1
        elif rating.iloc[i, 1] == '买入':
            rating.iloc[i, 1] = 2
        else:
            rating.iloc[i, 1] = 0
    '''if the report is released on a non-trading day, push it back to the last trading day'''
    rating = push_back_to_trading_day(rating, df)

    '''after push_back_to_trading_day, some reports may be stacked on the same day, calculate average value'''
    rating_ = pd.DataFrame()
    for i in set(rating['trade_date']):
        average = rating[rating['trade_date'] == i]['rating'].mean()
        rating_ = pd.concat([rating_, pd.DataFrame({'trade_date': [i], 'rating': [average]})])
    rating_ = rating_.sort_values(by='trade_date')
    rating = rating_
    return rating


def get_forecast_price(ts_code, df):
    ts_code_sp = ts_code[:6]
    '''read raw data'''
    forecast_price = pd.read_csv(r'./data/forecast/price.csv', low_memory=False).drop([0, 1])
    '''select lines with respect to target stock'''
    forecast_price = forecast_price[forecast_price['Stkcd'] == int(ts_code_sp)]
    ''''DeclareDate' is the date of declaration date of that report'''
    ''''ObjectPriceMin' is analysts' expected minimum price'''
    forecast_price = forecast_price[['DeclareDate', 'ObjectPriceMin']].copy()
    forecast_price = forecast_price.rename(columns={'DeclareDate': 'trade_date', 'ObjectPriceMin': 'forecast_price'})
    forecast_price['trade_date'] = pd.to_datetime(forecast_price['trade_date'])

    forecast_price = forecast_price.sort_values('trade_date')
    forecast_price = push_back_to_trading_day(forecast_price, df)

    '''after push_back_to_trading_day, some reports may be stacked on the same day, calculate average value'''
    forecast_price_ = pd.DataFrame()
    for i in set(forecast_price['trade_date']):
        ave_price = forecast_price[forecast_price['trade_date'] == i]['forecast_price'].mean()
        forecast_price_ = pd.concat([forecast_price_, pd.DataFrame({'trade_date': [i], 'forecast_price': [ave_price]})])
    forecast_price_ = forecast_price_.sort_values(by='trade_date')
    forecast_price = forecast_price_

    '''some outlier'''
    forecast_price['forecast_price'] = [i if i < 500
                                        else forecast_price['forecast_price'].mean() +
                                        forecast_price['forecast_price'].std()
                                        for i in forecast_price['forecast_price']]
    return forecast_price


def generate_sentiment(ts_code, df):
    ts_code_sp = ts_code[:6]
    '''read raw data'''
    news = pd.read_csv(f'./data/news/news_{ts_code_sp}.csv', encoding='utf-8')
    '''make date structure consistent'''
    news = news.rename(columns={'DeclareDate': 'trade_date'})
    news['trade_date'] = pd.to_datetime(news['trade_date'])
    news['trade_date'] = [i.date() for i in news['trade_date']]
    '''select the news data in research period'''
    news = news[news['trade_date'] <= datetime.date(2021, 12, 31)]
    news = news[news['trade_date'] >= datetime.date(2013, 6, 1)]

    '''calculate sentiment'''
    c = Cemotion()
    news['sentiment'] = [c.predict(i) for i in news['NewsContent']]

    '''
    Cemotion returns sentiment between 0 and 1, the larger, the more positive
    in this dataset, rare near 0.5 sentiment score
    label 1 for good news, 0 for bad news
    '''
    news['sentiment'] = np.where(news['sentiment'] > 0.5, 1, -1)

    '''directly sum up the sentiment scores, if one day has multiple pieces of news'''
    sentiment = pd.DataFrame()
    for i in set(news['trade_date']):
        sen = news[news['trade_date'] == i]['sentiment'].sum()
        sentiment = pd.concat([sentiment, pd.DataFrame({'trade_date': [i], 'sentiment': [sen]})])
    sentiment = sentiment.sort_values(by='trade_date')

    sentiment = push_back_to_trading_day(sentiment, df)
    '''after push_back_to_trading_day, some reports may be stacked on the same day, sum them up'''
    sentiment_ = pd.DataFrame()
    for i in set(sentiment['trade_date']):
        sen = sentiment[sentiment['trade_date'] == i]['sentiment'].sum()
        sentiment_ = pd.concat([sentiment_, pd.DataFrame({'trade_date': [i], 'sentiment': [sen]})])
    sentiment_ = sentiment_.sort_values(by='trade_date')

    return sentiment_
