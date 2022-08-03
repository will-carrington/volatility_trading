#volatility trading - this script not serve as financial advice, proceed at your OWN risk

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import requests
import time
import bs4

def flow2blo(stock,time_back, interval_per, target_band, interval_back):
  #flow2blo('aapl','20d','10m', 5, 15)
  target_band = target_band
  bol = interval_back
  rsi_ = interval_back
  ewm_ = interval_back
  b_devs = 2
  data = pd.DataFrame(yf.download(tickers = stock, period = time_back, interval = interval_per)).reset_index().reset_index()
  data['pchange'] = (data['Open']-data['Close'])/data['Open']*100

#metrics taken from investopedia (most popular technical indicators)
  five_int = []
  ten_int = [] 
  fft_int = [] 
  ffive = []
  tfive = [] 
  bolinger = [] 
  bolinger_2x = [] 
  bolinger_2d = [] 
  rsi = []
  rsi2 = []
  rsi3 = []
  macd = [] 
  macd2 = [] 
  macd3 = []
  obv = [] 
  obv1 = []
  obv2 = []
  ema_s = [] 
  ema_a = []
  ema_s1 = [] 
  ema_a1 = []
  ema_s2 = [] 
  ema_a2 = []
  ema_s3 = [] 
  ema_a3 = []
  avgRSI = []
  rsiBol = []
  rsiEMS = []
  emaMac = []
  macffIve = []
  jumboAvg = []
  jumboStd = []
  yahooB = []
  yahooTDY = []
  yahooDA = []
  yahooDY = []
  yahooPR = []
  yahooMV = []
  yahooB1 = []
  yahooTDY1 = []
  yahooDA1 = []
  yahooDY1 = []
  yahooPR1 = []
  yahooMV1 = []
  yahoo = yf.Ticker(stock)

  for x in data['index']:
    five_int.append(np.average(data['Close'].iloc[x-5:x]))
    ten_int.append(np.average(data['Close'].iloc[x-10:x]))
    fft_int.append(np.average(data['Close'].iloc[x-15:x]))
    tfive.append(np.average(data['Close'].iloc[x-25:x]))
    ffive.append(np.average(data['Close'].iloc[x-50:x]))

    bolinger.append(np.average((data['Close'].iloc[x-bol:x]+data['Low'].iloc[x-bol:x]+data['High'].iloc[x-bol:x])/3)+b_devs*np.std((data['Close'].iloc[x-bol:x]+data['Low'].iloc[x-bol:x]+data['High'].iloc[x-bol:x])/3))
    bolinger_2d.append(np.average((data['Close'].iloc[x-round(bol/2):x]+data['Low'].iloc[x-round(bol/2):x]+data['High'].iloc[x-round(bol/2):x])/3)+b_devs*np.std((data['Close'].iloc[x-round(bol/2):x]+data['Low'].iloc[x-round(bol/2):x]+data['High'].iloc[x-round(bol/2):x])/3))
    bolinger_2x.append(np.average((data['Close'].iloc[x-round(bol/3):x]+data['Low'].iloc[x-round(bol/3):x]+data['High'].iloc[x-round(bol/3):x])/3)+b_devs*np.std((data['Close'].iloc[x-round(bol/3):x]+data['Low'].iloc[x-round(bol/3):x]+data['High'].iloc[x-round(bol/3):x])/3))

    rsi.append(100-(100/(1+np.average(data['pchange'].iloc[x-rsi_:x][data['pchange'].iloc[x-rsi_:x]>=0])/np.average(data['pchange'].iloc[x-rsi_:x][data['pchange'].iloc[x-rsi_:x]<=0]))))
    rsi2.append(100-(100/(1+np.average(data['pchange'].iloc[x-round(rsi_/2):x][data['pchange'].iloc[x-round(rsi_/2):x]>=0])/np.average(data['pchange'].iloc[x-round(rsi_/2):x][data['pchange'].iloc[x-round(rsi_/2):x]<=0]))))
    rsi3.append(100-(100/(1+np.average(data['pchange'].iloc[x-round(rsi_/3):x][data['pchange'].iloc[x-round(rsi_/3):x]>=0])/np.average(data['pchange'].iloc[x-round(rsi_/3):x][data['pchange'].iloc[x-round(rsi_/3):x]<=0]))))

    macd.append(np.average(data['Close'].iloc[x-12:x]) - np.average(data['Close'].iloc[x-26:x]))
    macd2.append(np.average(data['Close'].iloc[x-12*2:x]) - np.average(data['Close'].iloc[x-50:x]))
   # macd3.append(np.average(data['Close'].iloc[x-12/3:x]) - np.average(data['Close'].iloc[x-27/3:x]))

    obv.append(np.average(data['Volume'].iloc[x-1:x]) + np.average(data['Volume'].iloc[x-2:x]))
    obv1.append(np.average(data['Volume'].iloc[x-5:x])/np.average(data['Volume'].iloc[x-5:x]))
    obv2.append(np.average(data['Volume'].iloc[x-10:x])/np.average(data['Volume'].iloc[x-10:x]))

    df = (data['High'].iloc[x-ewm_:x]+data['Low'].iloc[x-ewm_:x]+data['Close'].iloc[x-ewm_:x])/3
    ema_a.append(np.average(df.ewm(com = .4).mean()))
    ema_s.append(np.std(df.ewm(com = .4).mean()))
    ema_a1.append(np.average(df.ewm(com = .8).mean()))
    ema_s1.append(np.std(df.ewm(com = .8).mean()))
    ema_a2.append(np.average(df.ewm(com = .6).mean()))
    ema_s2.append(np.std(df.ewm(com = .6).mean()))
    ema_a3.append(np.average(df.ewm(com = .2).mean()))
    ema_s3.append(np.std(df.ewm(com = .2).mean()))
  
    avgRSI.append(rsi[x]/tfive[x])
    rsiBol.append(rsi[x]/bolinger[x])
    rsiEMS.append(rsi[x]/ema_s[x])
    emaMac.append(ema_a[x]/macd[x])
    macffIve.append(macd[x]/ffive[x])
    try:
        jumboStd.append(np.std(avgRSI,rsiBol,rsiEMS,emaMac,macffIve))
        jumboAvg.append(np.average(avgRSI,rsiBol,rsiEMS,emaMac,macffIve))
        yahooB.append(yahoo.info['beta']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooTDY.append(yahoo.info['trailingAnnualDividendYield']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooDA.append(yahoo.info['twoHundredDayAverage']*abs(jumboAvg[x-1]-jumboAvg[x]))
        #final['morningStarRiskRating'] = yahoo.info['morningStarRiskRating']
        yahooDY.append(yahoo.info['dividendYield']*abs(jumboAvg[x-1]-jumboAvg[x]))
        #final['shortRatio'] = yahoo.info['shortRatio']
        #final['sharesShortPreviousMonthDate'] = yahoo.info['sharesShortPreviousMonthDate']
        #final['volume24Hr'] = yahoo.info['volume24Hr']
        yahooPR.append(yahoo.info['payoutRatio']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooMV.append(yahoo.info['regularMarketVolume']*abs(jumboAvg[x-1]-jumboAvg[x]))
    except:
        yahooB.append(0)
        yahooTDY.append(0)
        yahooDA.append(0)
        yahooDY.append(0)
        yahooPR.append(0)
        yahooMV.append(0)

    #final['openInterest'] = yahoo.info['openInterest']
    #final['circulatingSupply'] = yahoo.info['circulatingSupply']
    #final['heldPercentInsiders'] = yahoo.info['heldPercentInsiders']
    #final['lastSplitFactor'] = yahoo.info['lastSplitFactor']
    try:
        yahooB1.append(yahoo.info['beta']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooTDY1.append(yahoo.info['trailingAnnualDividendYield']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooDA1.append(yahoo.info['twoHundredDayAverage']*abs(jumboAvg[x-10]-jumboAvg[x]))
        #final['morningStarRiskRating'] = yahoo.info['morningStarRiskRating']
        yahooDY1.append(yahoo.info['dividendYield']*abs(jumboAvg[x-10]-jumboAvg[x]))
        #final['shortRatio'] = yahoo.info['shortRatio']
        #final['sharesShortPreviousMonthDate'] = yahoo.info['sharesShortPreviousMonthDate']
        #final['volume24Hr'] = yahoo.info['volume24Hr']
        yahooPR1.append(yahoo.info['payoutRatio']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooMV1.append(yahoo.info['regularMarketVolume']*abs(jumboAvg[x-10]-jumboAvg[x]))
    except:
        yahooB1.append(0)
        yahooTDY1.append(0)
        yahooDA1.append(0)
        yahooDY1.append(0)
        yahooPR1.append(0)
        yahooMV1.append(0)



  final_t = pd.DataFrame()

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()

  final_t['beta'] = yahooB
  final_t['trailing_div_tield'] = yahooTDY
  final_t['twoHundoDayAvg']= yahooDA
  final_t['dividendYield'] = yahooDY
  final_t['payoutRatio'] = yahooPR
  final_t['marketvolume'] = yahooMV

  final_t['beta1'] = yahooB1
  final_t['trailing_div_tield1'] = yahooTDY1
  final_t['twoHundoDayAvg1']= yahooDA1
  final_t['dividendYield1'] = yahooDY1
  final_t['payoutRatio1'] = yahooPR1
  final_t['marketvolume1'] = yahooMV1

  
  final_t['5_sma'] =five_int
  final_t['10_sma'] =ten_int
  final_t['15_sma'] =fft_int
  final_t['25_sma'] = tfive
  final_t['50_sma'] = ffive
  final_t['bolinger'] = bolinger
  final_t['bolinger1'] = bolinger_2d
  final_t['bolinger2'] = bolinger_2x
  final_t['rsi1'] = rsi
  final_t['rsi2'] = rsi2
  final_t['rsi3'] = rsi3
  final_t['macd'] = macd
  final_t['macd2'] = macd2
  #final_t['macd3'] = macd3)
  final_t['obv'] =obv
  final_t['obv1'] =obv1
  final_t['obv2'] =obv2
  final_t['ema_std'] = ema_s
  final_t['ema_avg'] = ema_a
  final_t['ema_avg1'] = ema_a1
  final_t['ema_std1'] = ema_s1
  final_t['ema_avg2'] = ema_a2
  final_t['ema_std2'] = ema_s2
  final_t['ema_avg3'] = ema_a3
  final_t['ema_std3'] = ema_s3
  #final_t['std_Jumbo'] =jumboStd
  #final_t['avg_Jumbo'] = jumboAvg
  
  target = []
  for x in data['pchange']:
    if x >= .5:
      target.append(1)
    else:
      target.append(0)
  final_t['target'] = target
  
  final_t = final_t[final_t['target']==1].append(final_t[final_t['target']==0].sample(len(final_t[final_t['target']==1]))).sample(len(final_t[final_t['target']==1])*2)
  
  return final_t

df = pd.DataFrame()
def clean_dataset(df):
  assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
  df.dropna(inplace=True)
  indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
  return df[indices_to_keep].astype(np.float64)

def random_forest(x,y):
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  clf=RandomForestClassifier(n_estimators=1000, max_features=11, bootstrap = True, warm_start = True)
  clf.fit(X_train,y_train)
  y_pred=clf.predict(X_test)
  print(y_pred)
  print('random forest score = '+str(metrics.accuracy_score(y_test, y_pred)))
  return str(metrics.accuracy_score(y_test, y_pred)), clf

def xgb_forest(x,y):
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  xgb_clf = XGBClassifier(n_estimators=1000)
  xgb_clf.fit(X_train, y_train)
  score = xgb_clf.score(X_test, y_test)
  print('XGB Score = '+str(score))
  return str(score), xgb_clf

def gradient_boosted_forest(x,y):
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  lr_list = [0.05]
  for learning_rate in lr_list:
      gb_clf = GradientBoostingClassifier(n_estimators=1000, max_features=11, warm_start = True, learning_rate = learning_rate)
      gb_clf.fit(X_train, y_train)
      print("Learning rate: ", learning_rate)
      print("Gradient Boosted Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
      print("Gradient Boosted Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
  return str(gb_clf.score(X_test, y_test)), gb_clf

def runIt(days_back,trading_interval,pchange,eye_back):
  df = flow2blo('MSFT',days_back,trading_interval, pchange, eye_back)
  df = df.append(flow2blo('AAPL',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('ATNM',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('SOL',days_back,trading_interval, pchange, eye_back))

  df = df.append(flow2blo('BTX',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('ALPP',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('GBOX',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('VVPR',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('FTEK',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('BTBT',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('OCGN',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('ICD',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('FCEL',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('APVO',days_back,trading_interval, pchange, eye_back))

  df = df.append(flow2blo('ICD',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('SM',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('KODK',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('SUP',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('MTDR',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('AR',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('LPI',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('FUBO',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('RYAM',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('NINE',days_back,trading_interval, pchange, eye_back))

  df = df.append(flow2blo('FET',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('BE',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('PRTY',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('DBD',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('RFP',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('OVV',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('OIS',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('WES',days_back,trading_interval, pchange, eye_back))
  df = df.append(flow2blo('FTCH',days_back,trading_interval, pchange, eye_back))
  
  print('training dataframe length: '+str(len(df)))

  
  df = df.sample(len(df))

  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  x = clean_dataset(df)
  y = x['target']
  df = scaler.fit(x.drop(columns = 'target'))
  x = pd.DataFrame(scaler.transform(x.drop(columns = 'target')))

  xc, clf = random_forest(x,y)
  xv, xgb_clf = xgb_forest(x,y)
  xg, gb_clf = gradient_boosted_forest(x,y)

  return xc, xv, xg, clf, xgb_clf, gb_clf

#step 1
forest, xgb , gb, clf, xgb_clf, gb_clf = runIt('7d','1m',.5,60)
#THIS IS THE TD AMERITRADE PART, plug in your own #s after making an account + developers account
ACCT_NUMBER = 
API_KEY = ''
CALLBACK_URL = 'http://localhost'

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from tda import auth, client
from webdriver_manager.chrome import ChromeDriverManager

def auth_func():

    token_path = 'token.pickle'
    try:
        c = auth.client_from_token_file(token_path, API_KEY)
    except FileNotFoundError:
        from selenium import webdriver
        with webdriver.Chrome(ChromeDriverManager().install()) as driver:
            c = auth.client_from_login_flow(
                driver, API_KEY, CALLBACK_URL, token_path)
    return c

c = auth_func()

from tda.orders.equities import equity_buy_market, equity_sell_market, equity_sell_limit, equity_buy_limit
from tda.orders.common import Duration, Session

#(c, buy/sell,shares,stock)
def place_order(c, order_type, shares, STOCK):
    if order_type == 'buy':
        order_spec = equity_buy_market(STOCK, shares).set_session(Session.NORMAL).set_duration(Duration.DAY).build()
        c.place_order(ACCT_NUMBER, order_spec)
    if order_type == 'sell':
        order_spec = equity_sell_market(STOCK, shares).set_session(
            Session.NORMAL).set_duration(Duration.DAY).build()
        c.place_order(ACCT_NUMBER, order_spec)

def place_limit_order(c,order_type,shares,STOCK, price):
    if order_type == 'buy':
        order_spec = equity_buy_limit(STOCK,1,price).set_session(Session.NORMAL).set_duration(Duration.DAY).build()
        c.place_order(ACCT_NUMBER,order_spec)
    if order_type == 'sell':
        order_spec = equity_sell_limit(STOCK,1,price).set_session(Session.NORMAL).set_duration(Duration.DAY).build()
        c.place_order(ACCT_NUMBER,order_spec)
    
def Credit_Sticks1(c, STOCK, ask_hold):
    r = c.get_quote(STOCK)
    assert r.status_code == 200, r.raise_for_status()
    y = r.json()
    bid_price = float(y[STOCK]['bidPrice'])
    bid_quantity = int(y[STOCK]['bidSize'])
    if ask_hold ==0:
        ask_hold = bid_price
    ask_price = float(y[STOCK]['askPrice'])
    ask_hold = ask_price
    ask_quantity = int(y[STOCK]['askSize'])
    price = (bid_price+((ask_price-bid_price)*bid_quantity/(ask_quantity+bid_quantity))+ask_price+((ask_price-bid_price)*ask_quantity/(ask_quantity+bid_quantity)))/2
    if price ==0:
        price = ask_price
    if bid_price ==0:
        bid_price = ask_price
    pchange = (ask_price-ask_hold)/ask_hold*100
    volume = bid_quantity+ask_quantity
    return bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold

def Credit_Sticks(c, STOCK):
    r = c.get_quote(STOCK)
    assert r.status_code == 200, r.raise_for_status()
    y = r.json()
    ask1 = float(y[STOCK]['bidPrice'])
    bid_price = float(y[STOCK]['bidPrice'])
    bid_quantity = int(y[STOCK]['bidSize'])
    ask_price = float(y[STOCK]['askPrice'])
    ask_quantity = int(y[STOCK]['askSize'])
    price = (bid_price+((ask_price-bid_price)*bid_quantity/(ask_quantity+bid_quantity))+ask_price+((ask_price-bid_price)*ask_quantity/(ask_quantity+bid_quantity)))/2

    pchange = (ask_price-ask1)/ask1*100
    volume = bid_quantity+ask_quantity
    return bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume



def stock_setup(STOCK):
  CS = {'bid_price':[],'bid_quantity':[],'ask_price':[],'ask_quantity':[],'price':[],'pchange':[],'volume':[]}
  import time
  for x in range(60):
        time.sleep(1)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume= Credit_Sticks(c, STOCK)
        CS['bid_price'].append(bid_price)
        CS['bid_quantity'].append(bid_quantity)
        CS['ask_price'].append(ask_price)
        CS['ask_quantity'].append(ask_quantity)
        CS['price'].append(price)
        CS['pchange'].append(pchange)
        CS['volume'].append(volume)
  return CS
  #flow2blo('aapl','20d','10m', 5, 15)

def flow2go_predictor(STOCK,target_band,interval_back, CS):
  import pandas as pd
  import numpy as np
  target_band = target_band
  bol = interval_back
  rsi_ = interval_back
  ewm_ = interval_back
  b_devs = 2
  data = pd.DataFrame(CS)

  
  data = data.reset_index()
  
  target_band = target_band
  bol = interval_back
  rsi_ = interval_back
  ewm_ = interval_back
  b_devs = 2

#metrics taken from investopedia (most popular technical indicators)
  five_int = []
  ten_int = [] 
  fft_int = [] 
  ffive = []
  tfive = [] 
  bolinger = [] 
  bolinger_2x = [] 
  bolinger_2d = [] 
  rsi = []
  rsi2 = []
  rsi3 = []
  macd = [] 
  macd2 = [] 
  macd3 = []
  obv = [] 
  obv1 = []
  obv2 = []
  ema_s = [] 
  ema_a = []
  ema_s1 = [] 
  ema_a1 = []
  ema_s2 = [] 
  ema_a2 = []
  ema_s3 = [] 
  ema_a3 = []
  avgRSI = []
  rsiBol = []
  rsiEMS = []
  emaMac = []
  macffIve = []
  jumboAvg = []
  jumboStd = []
  yahooB = []
  yahooTDY = []
  yahooDA = []
  yahooDY = []
  yahooPR = []
  yahooMV = []
  yahooB1 = []
  yahooTDY1 = []
  yahooDA1 = []
  yahooDY1 = []
  yahooPR1 = []
  yahooMV1 = []
  yahoo = yf.Ticker(STOCK)
  
  data['Close'] = CS['ask_price']
  data['Open'] = CS['ask_price']
  data['pchange'] = CS['pchange']*100
  data['Volume'] = CS['volume']

  for x in data['index']:
    five_int.append(np.average(data['Close'].iloc[x-5:x]))
    ten_int.append(np.average(data['Close'].iloc[x-10:x]))
    fft_int.append(np.average(data['Close'].iloc[x-15:x]))
    tfive.append(np.average(data['Close'].iloc[x-25:x]))
    ffive.append(np.average(data['Close'].iloc[x-50:x]))

    bolinger.append(np.average((data['Close'].iloc[x-bol:x]+data['Low'].iloc[x-bol:x]+data['High'].iloc[x-bol:x])/3)+b_devs*np.std((data['Close'].iloc[x-bol:x]+data['Low'].iloc[x-bol:x]+data['High'].iloc[x-bol:x])/3))
    bolinger_2d.append(np.average((data['Close'].iloc[x-round(bol/2):x]+data['Low'].iloc[x-round(bol/2):x]+data['High'].iloc[x-round(bol/2):x])/3)+b_devs*np.std((data['Close'].iloc[x-round(bol/2):x]+data['Low'].iloc[x-round(bol/2):x]+data['High'].iloc[x-round(bol/2):x])/3))
    bolinger_2x.append(np.average((data['Close'].iloc[x-round(bol/3):x]+data['Low'].iloc[x-round(bol/3):x]+data['High'].iloc[x-round(bol/3):x])/3)+b_devs*np.std((data['Close'].iloc[x-round(bol/3):x]+data['Low'].iloc[x-round(bol/3):x]+data['High'].iloc[x-round(bol/3):x])/3))

    rsi.append(100-(100/(1+np.average(data['pchange'].iloc[x-rsi_:x][data['pchange'].iloc[x-rsi_:x]>=0])/np.average(data['pchange'].iloc[x-rsi_:x][data['pchange'].iloc[x-rsi_:x]<=0]))))
    rsi2.append(100-(100/(1+np.average(data['pchange'].iloc[x-round(rsi_/2):x][data['pchange'].iloc[x-round(rsi_/2):x]>=0])/np.average(data['pchange'].iloc[x-round(rsi_/2):x][data['pchange'].iloc[x-round(rsi_/2):x]<=0]))))
    rsi3.append(100-(100/(1+np.average(data['pchange'].iloc[x-round(rsi_/3):x][data['pchange'].iloc[x-round(rsi_/3):x]>=0])/np.average(data['pchange'].iloc[x-round(rsi_/3):x][data['pchange'].iloc[x-round(rsi_/3):x]<=0]))))

    macd.append(np.average(data['Close'].iloc[x-12:x]) - np.average(data['Close'].iloc[x-26:x]))
    macd2.append(np.average(data['Close'].iloc[x-12*2:x]) - np.average(data['Close'].iloc[x-50:x]))
  #  macd3.append(np.average(data['Close'].iloc[x-12/3:x]) - np.average(data['Close'].iloc[x-27/3:x]))

    obv.append(np.average(data['Volume'].iloc[x-1:x]) + np.average(data['Volume'].iloc[x-2:x]))
    obv1.append(np.average(data['Volume'].iloc[x-5:x])/np.average(data['Volume'].iloc[x-5:x]))
    obv2.append(np.average(data['Volume'].iloc[x-10:x])/np.average(data['Volume'].iloc[x-10:x]))

    
    df = (data['High'].iloc[x-ewm_:x]+data['Low'].iloc[x-ewm_:x]+data['Close'].iloc[x-ewm_:x])/3
    ema_a.append(np.average(df.ewm(com = .4).mean()))
    ema_s.append(np.std(df.ewm(com = .4).mean()))
    ema_a1.append(np.average(df.ewm(com = .8).mean()))
    ema_s1.append(np.std(df.ewm(com = .8).mean()))
    ema_a2.append(np.average(df.ewm(com = .6).mean()))
    ema_s2.append(np.std(df.ewm(com = .6).mean()))
    ema_a3.append(np.average(df.ewm(com = .2).mean()))
    ema_s3.append(np.std(df.ewm(com = .2).mean()))
  
    avgRSI.append(rsi[x]/tfive[x])
    rsiBol.append(rsi[x]/bolinger[x])
    rsiEMS.append(rsi[x]/ema_s[x])
    emaMac.append(ema_a[x]/macd[x])
    macffIve.append(macd[x]/ffive[x])
    try:
        jumboStd.append(np.std(avgRSI,rsiBol,rsiEMS,emaMac,macffIve))
        jumboAvg.append(np.average(avgRSI,rsiBol,rsiEMS,emaMac,macffIve))
        yahooB.append(yahoo.info['beta']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooTDY.append(yahoo.info['trailingAnnualDividendYield']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooDA.append(yahoo.info['twoHundredDayAverage']*abs(jumboAvg[x-1]-jumboAvg[x]))
        #final['morningStarRiskRating'] = yahoo.info['morningStarRiskRating']
        yahooDY.append(yahoo.info['dividendYield']*abs(jumboAvg[x-1]-jumboAvg[x]))
        #final['shortRatio'] = yahoo.info['shortRatio']
        #final['sharesShortPreviousMonthDate'] = yahoo.info['sharesShortPreviousMonthDate']
        #final['volume24Hr'] = yahoo.info['volume24Hr']
        yahooPR.append(yahoo.info['payoutRatio']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooMV.append(yahoo.info['regularMarketVolume']*abs(jumboAvg[x-1]-jumboAvg[x]))
    except:
        yahooB.append(0)
        yahooTDY.append(0)
        yahooDA.append(0)
        yahooDY.append(0)
        yahooPR.append(0)
        yahooMV.append(0)

    #final['openInterest'] = yahoo.info['openInterest']
    #final['circulatingSupply'] = yahoo.info['circulatingSupply']
    #final['heldPercentInsiders'] = yahoo.info['heldPercentInsiders']
    #final['lastSplitFactor'] = yahoo.info['lastSplitFactor']
    try:
        yahooB1.append(yahoo.info['beta']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooTDY1.append(yahoo.info['trailingAnnualDividendYield']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooDA1.append(yahoo.info['twoHundredDayAverage']*abs(jumboAvg[x-10]-jumboAvg[x]))
        #final['morningStarRiskRating'] = yahoo.info['morningStarRiskRating']
        yahooDY1.append(yahoo.info['dividendYield']*abs(jumboAvg[x-10]-jumboAvg[x]))
        #final['shortRatio'] = yahoo.info['shortRatio']
        #final['sharesShortPreviousMonthDate'] = yahoo.info['sharesShortPreviousMonthDate']
        #final['volume24Hr'] = yahoo.info['volume24Hr']
        yahooPR1.append(yahoo.info['payoutRatio']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooMV1.append(yahoo.info['regularMarketVolume']*abs(jumboAvg[x-10]-jumboAvg[x]))
    except:
        yahooB1.append(0)
        yahooTDY1.append(0)
        yahooDA1.append(0)
        yahooDY1.append(0)
        yahooPR1.append(0)
        yahooMV1.append(0)




  final_t = pd.DataFrame()
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  

  final_t['beta'] = yahooB
  final_t['trailing_div_tield'] = yahooTDY
  final_t['twoHundoDayAvg']= yahooDA
  final_t['dividendYield'] = yahooDY
  final_t['payoutRatio'] = yahooPR
  final_t['marketvolume'] = yahooMV

  final_t['beta1'] = yahooB1
  final_t['trailing_div_tield1'] = yahooTDY1
  final_t['twoHundoDayAvg1']= yahooDA1
  final_t['dividendYield1'] = yahooDY1
  final_t['payoutRatio1'] = yahooPR1
  final_t['marketvolume1'] = yahooMV1

  
  final_t['5_sma'] =five_int
  final_t['10_sma'] =ten_int
  final_t['15_sma'] =fft_int
  final_t['25_sma'] = tfive
  final_t['50_sma'] = ffive
  final_t['bolinger'] = bolinger
  final_t['bolinger1'] = bolinger_2d
  final_t['bolinger2'] = bolinger_2x
  final_t['rsi1'] = rsi
  final_t['rsi2'] = rsi2
  final_t['rsi3'] = rsi3
  final_t['macd'] = macd
  final_t['macd2'] = macd2
  #final_t['macd3'] = macd3)
  final_t['obv'] =obv
  final_t['obv1'] =obv1
  final_t['obv2'] =obv2
  final_t['ema_std'] = ema_s
  final_t['ema_avg'] = ema_a
  final_t['ema_avg1'] = ema_a1
  final_t['ema_std1'] = ema_s1
  final_t['ema_avg2'] = ema_a2
  final_t['ema_std2'] = ema_s2
  final_t['ema_avg3'] = ema_a3
  final_t['ema_std3'] = ema_s3
#  final_t['std_Jumbo'] =jumboStd
 # final_t['avg_Jumbo'] = jumboAvg
    
  return final_t
  
  
def credit_sticks(num_minutes, STOCK1,STOCK2,STOCK3,STOCK4,STOCK5,STOCK6,STOCK7,STOCK8,STOCK9,STOCK10):
    CS1 = stock_setup(STOCK1)
    CS2 = stock_setup(STOCK2)
    CS3 = stock_setup(STOCK3)
    CS4 = stock_setup(STOCK4)
    CS5 = stock_setup(STOCK5)
    CS6 = stock_setup(STOCK6)
    CS7 = stock_setup(STOCK7)
    CS8 = stock_setup(STOCK8)
    CS9 = stock_setup(STOCK9)
    CS10 = stock_setup(STOCK10)


    for xx in range(num_minutes):
        #selling back
        final_t1 = flow2go_predictor(STOCK1,.5, 60, CS1).fillna(0)
        final_t2 = flow2go_predictor(STOCK2,.5, 60, CS2).fillna(0)
        final_t3 = flow2go_predictor(STOCK3,.5, 60, CS3).fillna(0)
        final_t4 = flow2go_predictor(STOCK4,.5, 60, CS4).fillna(0)
        final_t5 = flow2go_predictor(STOCK5,.5, 60, CS5).fillna(0)
        final_t6 = flow2go_predictor(STOCK6,.5, 60, CS6).fillna(0)
        final_t7 = flow2go_predictor(STOCK7,.5, 60, CS7).fillna(0)
        final_t8 = flow2go_predictor(STOCK8,.5, 60, CS8).fillna(0)
        final_t9 = flow2go_predictor(STOCK9,.5, 60, CS9).fillna(0)
        final_t10 = flow2go_predictor(STOCK10,.5, 60, CS10).fillna(0)
        #1: buy, 0: sell, 2: hold
        #buying
        
        x1 = int(clf.predict_proba(final_t1[-1])[:, 1])
        x11 = int(xgb_clf.predict_proba(final_t1[-1])[:, 1])
        x111 = int(gb_clf.predict_proba(final_t1[-1])[:, 1])
        print(x1)
        x2 = int(clf.predict_proba(final_t2[-1])[:, 1])
        x22 = int(xgb_clf.predict_proba(final_t2[-1])[:, 1])
        x222 = int(gb_clf.predict_proba(final_t2[-1])[:, 1])
        print(x2)
        x3 = int(clf.predict_proba(final_t3[-1])[:, 1])
        x33 = int(xgb_clf.predict_proba(final_t3[-1])[:, 1])
        x333 = int(gb_clf.predict_proba(final_t3[-1])[:, 1])
        print(x3)
        x4 = int(clf.predict_proba(final_t4[-1])[:, 1])
        x44 = int(xgb_clf.predict_proba(final_t4[-1])[:, 1])
        x444 = int(gb_clf.predict_proba(final_t4[-1])[:, 1])
        
        x5 = int(clf.predict_proba(final_t5[-1])[:, 1])
        x55 = int(xgb_clf.predict_proba(final_t5[-1])[:, 1])
        x555 = int(gb_clf.predict_proba(final_t5[-1])[:, 1])
        
        x6 = int(clf.predict_proba(final_t6[-1])[:, 1])
        x66 = int(xgb_clf.predict_proba(final_t6[-1])[:, 1])
        x666 = int(gb_clf.predict_proba(final_t6[-1])[:, 1])
        
        x7 = int(clf.predict_proba(final_t7[-1])[:, 1])
        x77 = int(xgb_clf.predict_proba(final_t7[-1])[:, 1])
        x777 = int(gb_clf.predict_proba(final_t7[-1])[:, 1])
        
        x8 = int(clf.predict_proba(final_t8[-1])[:, 1])
        x88 = int(xgb_clf.predict_proba(final_t8[-1])[:, 1])
        x888 = int(gb_clf.predict_proba(final_t8[-1])[:, 1])
        
        x9 = int(clf.predict_proba(final_t9[-1])[:, 1])
        x99 = int(xgb_clf.predict_proba(final_t9[-1])[:, 1])
        x999 = int(gb_clf.predict_proba(final_t9[-1])[:, 1])
        
        x10 = int(clf.predict_proba(final_t10[-1])[:, 1])
        x100 = int(xgb_clf.predict_proba(final_t10[-1])[:, 1])
        x1000 = int(gb_clf.predict_proba(final_t10[-1])[:, 1])
        
        jj10 = np.average(x10,x100,x1000)  
        jj1 = np.average(x1,x11,x111)       
        jj2 = np.average(x2,x22,x222)       
        jj3 = np.average(x3,x33,x333)       
        jj4 = np.average(x4,x44,x444)       
        jj5 = np.average(x5,x55,x555)       
        jj6 = np.average(x6,x66,x666)       
        jj7 = np.average(x7,x77,x777)       
        jj8 = np.average(x8,x88,x888)       
        jj9 = np.average(x9,x99,x999)  
        print(max(jj1,jj2,jj3,jj4,jj5,jj6,jj7,jj8,jj9,jj10))
        
        t = 0
        if jj10 >=.55:
            place_order(c,'buy',round(10),STOCK10)
            t = t+1
            time.sleep(1)
            
        if jj1 >=.55:
            place_order(c,'buy',round(20),STOCK1)
            t = t+1
            time.sleep(1)

        if jj2 >=.55:
            place_order(c,'buy',round(20),STOCK2)
            t = t+1
            time.sleep(1)

        if jj3 >=.55:
            place_order(c,'buy',round(20),STOCK3)
            t = t+1
            time.sleep(1)

        if jj4 >=.55:
            place_order(c,'buy',round(20),STOCK4)
            t = t+1
            time.sleep(1)

        if jj5 >=.55:
            place_order(c,'buy',round(20),STOCK5)
            t = t+1
            time.sleep(1)
        if jj6 >=.55:
            place_order(c,'buy',round(20),STOCK6)
            t = t+1
            time.sleep(1)
        if jj7 >=.55:
            place_order(c,'buy',round(20),STOCK7)
            t = t+1
            time.sleep(1)
        if jj8 >=.55:
            place_order(c,'buy',20,STOCK8)
            t = t+1
            time.sleep(1)
        if jj9>=.55:
            place_order(c,'buy',10,STOCK9)
            t = t+1
            time.sleep(1)
        print('# of stocks bought = '+str(t))
            
        time.sleep(40-t)

        if jj10 >=.55:
            place_order(c,'sell',round(10),STOCK10)
            t = t+1
            time.sleep(1)
            
        if jj1 >=.55:
            place_order(c,'sell',round(20),STOCK1)
            t = t+1
            time.sleep(1)

        if jj2 >=.55:
            # place_order(c,'sell',round(20),STOCK2)
            t = t+1
            time.sleep(1)

        if jj3 >=.55:
            place_order(c,'sell',round(20),STOCK3)
            t = t+1
            time.sleep(1)

        if jj4 >=.55:
            place_order(c,'sell',round(20),STOCK4)
            t = t+1
            time.sleep(1)

        if jj5 >=.55:
            place_order(c,'sell',round(20),STOCK5)
            t = t+1
            time.sleep(1)
        if jj6 >=.55:
            place_order(c,'sell',round(20),STOCK6)
            t = t+1
            time.sleep(1)
        if jj7 >=.55:
            place_order(c,'sell',round(20),STOCK7)
            t = t+1
            time.sleep(1)
        if jj8 >=.55:
            place_order(c,'sell',20,STOCK8)
            t = t+1
            time.sleep(1)
        if jj9>=.55:
            place_order(c,'sell',10,STOCK9)
            t = t+1
            time.sleep(1)
        if xx ==0:
            ask_hold = 0
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Sticks1(c, STOCK1, ask_hold)
        CS1['bid_price'].append(bid_price)
        CS1['bid_quantity'].append(bid_quantity)
        CS1['ask_price'].append(ask_price)
        CS1['ask_quantity'].append(ask_quantity)
        CS1['price'].append(price)
        CS1['pchange'].append(pchange)
        CS1['volume'].append(volume)

        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Sticks1(c, STOCK2, ask_hold)
        CS2['bid_price'].append(bid_price)
        CS2['bid_quantity'].append(bid_quantity)
        CS2['ask_price'].append(ask_price)
        CS2['ask_quantity'].append(ask_quantity)
        CS2['price'].append(price)
        CS2['pchange'].append(pchange)
        CS2['volume'].append(volume)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume , ask_hold= Credit_Sticks1(c, STOCK3, ask_hold)
        CS3['bid_price'].append(bid_price)
        CS3['bid_quantity'].append(bid_quantity)
        CS3['ask_price'].append(ask_price)
        CS3['ask_quantity'].append(ask_quantity)
        CS3['price'].append(price)
        CS3['pchange'].append(pchange)
        CS3['volume'].append(volume)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Sticks1(c, STOCK4, ask_hold)
        CS4['bid_price'].append(bid_price)
        CS4['bid_quantity'].append(bid_quantity)
        CS4['ask_price'].append(ask_price)
        CS4['ask_quantity'].append(ask_quantity)
        CS4['price'].append(price)
        CS4['pchange'].append(pchange)
        CS4['volume'].append(volume)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Sticks1(c, STOCK5, ask_hold)
        CS5['bid_price'].append(bid_price)
        CS5['bid_quantity'].append(bid_quantity)
        CS5['ask_price'].append(ask_price)
        CS5['ask_quantity'].append(ask_quantity)
        CS5['price'].append(price)
        CS5['pchange'].append(pchange)
        CS5['volume'].append(volume)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Sticks1(c, STOCK6, ask_hold)
        CS6['bid_price'].append(bid_price)
        CS6['bid_quantity'].append(bid_quantity)
        CS6['ask_price'].append(ask_price)
        CS6['ask_quantity'].append(ask_quantity)
        CS6['price'].append(price)
        CS6['pchange'].append(pchange)
        CS6['volume'].append(volume)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume , ask_hold= Credit_Sticks1(c, STOCK7, ask_hold)
        CS7['bid_price'].append(bid_price)
        CS7['bid_quantity'].append(bid_quantity)
        CS7['ask_price'].append(ask_price)
        CS7['ask_quantity'].append(ask_quantity)
        CS7['price'].append(price)
        CS7['pchange'].append(pchange)
        CS7['volume'].append(volume)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Sticks1(c, STOCK8, ask_hold)
        CS8['bid_price'].append(bid_price)
        CS8['bid_quantity'].append(bid_quantity)
        CS8['ask_price'].append(ask_price)
        CS8['ask_quantity'].append(ask_quantity)
        CS8['price'].append(price)
        CS8['pchange'].append(pchange)
        CS8['volume'].append(volume)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Sticks1(c, STOCK9, ask_hold)
        CS9['bid_price'].append(bid_price)
        CS9['bid_quantity'].append(bid_quantity)
        CS9['ask_price'].append(ask_price)
        CS9['ask_quantity'].append(ask_quantity)
        CS9['price'].append(price)
        CS9['pchange'].append(pchange)
        CS9['volume'].append(volume)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Sticks1(c, STOCK10, ask_hold)
        CS10['bid_price'].append(bid_price)
        CS10['bid_quantity'].append(bid_quantity)
        CS10['ask_price'].append(ask_price)
        CS10['ask_quantity'].append(ask_quantity)
        CS10['price'].append(price)
        CS10['pchange'].append(pchange)
        CS10['volume'].append(volume)



#minutes in a trading day: 6.5*60 = 690 --> RUN 45 minutes pre market!!

#stocks <$1: CDTX: therapeutics, UBX: biotech, CRTX (pharma), INFI (pharma), 
#AGTC (genetic tech), FBIO (biotech), SYRS (pharma), SYBX (SYBX)
#, SIEN (medical sales), CABA (biotech), GSV (mining), CUBXF (cubic farming systems), 
#POFCY (energy), CYBN (psychedelic tehrapetuics), APTX (biopharma), SLHG, UPH (digital health)
# CELTF (mining)

credit_sticks(200,'BTX','APVO','ALPP','GBOX','VVPR','FTEK','BTBT','OCGN', 'ICD','FCEL')

#OMGA, TEDU, NOK, + SAI --> 