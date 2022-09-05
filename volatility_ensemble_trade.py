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

def clean_dataset(df):
  assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
  df.dropna(inplace=True)
  indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
  return df[indices_to_keep].astype(np.float64)


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

def f2g(stock,time_back, interval_per, target_band, interval_back):
  target_band = target_band
  bol = interval_back
  rsi_ = interval_back
  ewm_ = interval_back
  b_devs = 2
  data = pd.DataFrame(yf.download(tickers = stock, period = time_back, interval = interval_per)).reset_index().reset_index()
  data['pchange'] = (data['Open']-data['Close'])/data['Open']*100

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
        yahooDY.append(yahoo.info['dividendYield']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooPR.append(yahoo.info['payoutRatio']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooMV.append(yahoo.info['regularMarketVolume']*abs(jumboAvg[x-1]-jumboAvg[x]))
    except:
        yahooB.append(0)
        yahooTDY.append(0)
        yahooDA.append(0)
        yahooDY.append(0)
        yahooPR.append(0)
        yahooMV.append(0)

    try:
        yahooB1.append(yahoo.info['beta']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooTDY1.append(yahoo.info['trailingAnnualDividendYield']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooDA1.append(yahoo.info['twoHundredDayAverage']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooDY1.append(yahoo.info['dividendYield']*abs(jumboAvg[x-10]-jumboAvg[x]))
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
  
  target = []
  for x in data['pchange']:
    if x >= .5:
      target.append(1)
    else:
      target.append(0)
  final_t['target'] = target
 
  final_t = final_t[final_t['target']==1].append(final_t[final_t['target']==0].sample(len(final_t[final_t['target']==1]), replace = True)).sample(len(final_t[final_t['target']==1])*2, replace = True)
 
    
  return final_t


df = pd.DataFrame()


def runIt(days_back,trading_interval,pchange,eye_back):
  df = f2g('MSFT',days_back,trading_interval, pchange, eye_back)
  df = df.append(f2g('AAPL',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('ADBE',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('META',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('AMZN',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('GOOG',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('NVDA',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('PYPL',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('TSLA',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('INTC',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('COST',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('TMUS',days_back,trading_interval, pchange, eye_back))

  df = df.append(f2g('TSM',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('ASML',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('ORCL',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('TXN',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('CRM',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('CSCO',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('AVGO',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('IBM',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('NOW',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('AMAT',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('ADI',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('UBER',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('EQIX',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('FIS',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('SNPS',days_back,trading_interval, pchange, eye_back))

  df = df.append(f2g('BTX',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('ALPP',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('GBOX',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('VVPR',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('FTEK',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('BTBT',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('OCGN',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('ICD',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('FCEL',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('APVO',days_back,trading_interval, pchange, eye_back))

  df = df.append(f2g('ICD',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('SM',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('KODK',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('SUP',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('MTDR',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('AR',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('LPI',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('FUBO',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('RYAM',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('NINE',days_back,trading_interval, pchange, eye_back))

  df = df.append(f2g('FET',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('BE',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('PRTY',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('DBD',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('RFP',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('OVV',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('OIS',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('WES',days_back,trading_interval, pchange, eye_back))
  df = df.append(f2g('FTCH',days_back,trading_interval, pchange, eye_back))
 
  print('training dataframe length: '+str(len(df)))

 
  df = df.sample(len(df), replace = True)

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


forest, xgb , gb, clf, xgb_clf, gb_clf = runIt('60d','15m',.01,15)
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
        
import random
def Credit_Swish1(c, STOCK, ask_hold):
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
        stock = STOCK
        stock = stock+'+stock+current+price'
        url = 'https://www.google.com/search?q='+stock
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36'}
        html = requests.get(url,headers = headers).text
        soup = bs4.BeautifulSoup(html, "html.parser")
        A = []
        for tag in soup.findAll('div',{'class':'PZPZlf'}):
            A.append(tag.findNext('span').text)
        price = float(A[3].replace(' USD',''))
    if bid_price ==0:
        bid_price = price
        ask_price = price + random.uniform(0,4)
    pchange = (ask_price-ask_hold)/ask_hold*100
    volume = bid_quantity+ask_quantity
    return bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold

def Credit_Swish(c, STOCK):
    r = c.get_quote(STOCK)
    assert r.status_code == 200, r.raise_for_status()
    y = r.json()
    ask1 = float(y[STOCK]['bidPrice'])
    bid_price = float(y[STOCK]['bidPrice'])
    bid_quantity = int(y[STOCK]['bidSize'])
    ask_price = float(y[STOCK]['askPrice'])
    ask_quantity = int(y[STOCK]['askSize'])
    price = (bid_price+((ask_price-bid_price)*bid_quantity/(ask_quantity+bid_quantity))+ask_price+((ask_price-bid_price)*ask_quantity/(ask_quantity+bid_quantity)))/2
    if price ==0:
        stock = STOCK
        stock = stock+'+stock+current+price'
        url = 'https://www.google.com/search?q='+stock
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36'}
        html = requests.get(url,headers = headers).text
        soup = bs4.BeautifulSoup(html, "html.parser")
        A = []
        for tag in soup.findAll('div',{'class':'PZPZlf'}):
            A.append(tag.findNext('span').text)
        price = float(A[3].replace(' USD',''))
    if bid_price ==0:
        bid_price = price
        ask_price = price + random.uniform(0,4)

    pchange = (ask_price-ask1)/ask1*100
    volume = bid_quantity+ask_quantity
    return bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume



def stock_setup(STOCK):
  CS = {'bid_price':[],'bid_quantity':[],'ask_price':[],'ask_quantity':[],'price':[],'pchange':[],'volume':[]}
  import time
  for x in range(15):
        time.sleep(random.uniform(3,10))
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume= Credit_Swish(c, STOCK)
        CS['bid_price'].append(bid_price)
        CS['bid_quantity'].append(bid_quantity)
        CS['ask_price'].append(ask_price)
        CS['ask_quantity'].append(ask_quantity)
        CS['price'].append(price)
        CS['pchange'].append(pchange)
        CS['volume'].append(volume)
  return CS

def flow2go_predictor(xx,STOCK,target_band,interval_back, CS):
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
  data = pd.DataFrame()
  data['Close'] = pd.DataFrame(CS['ask_price'])+pd.DataFrame(CS['pchange'])
  data['Open'] = pd.DataFrame(CS['ask_price'])-pd.DataFrame(CS['pchange'])

  data['Low'] = CS['bid_price']
  data['High'] = CS['ask_price']

  data['pchange'] = CS['pchange']
  data['Volume'] = CS['volume']
  data = data.reset_index()

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
        yahooDY.append(yahoo.info['dividendYield']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooPR.append(yahoo.info['payoutRatio']*abs(jumboAvg[x-1]-jumboAvg[x]))
        yahooMV.append(yahoo.info['regularMarketVolume']*abs(jumboAvg[x-1]-jumboAvg[x]))
    except:
        yahooB.append(0)
        yahooTDY.append(0)
        yahooDA.append(0)
        yahooDY.append(0)
        yahooPR.append(0)
        yahooMV.append(0)

    try:
        yahooB1.append(yahoo.info['beta']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooTDY1.append(yahoo.info['trailingAnnualDividendYield']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooDA1.append(yahoo.info['twoHundredDayAverage']*abs(jumboAvg[x-10]-jumboAvg[x]))
        yahooDY1.append(yahoo.info['dividendYield']*abs(jumboAvg[x-10]-jumboAvg[x]))
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

  if xx == 0:
      x = final_t 
  else:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x = clean_dataset(final_t)
    df = scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x))
     
  return x
avg_prob = []
avg_probGB = []
avg_probXGB = []


def printer(STOCK1,CS1,xx, bought):   
    if bought == str(xx-1)+STOCK1:
        place_order(c, 'sell', 2, STOCK1)
    time.sleep(3)
    final_t1 = flow2go_predictor(xx,STOCK1,.01, 15, CS1).fillna(0)
    x1 = float(clf.predict_proba(final_t1)[-1][1])
    print(STOCK1+' Random Forest probability of increasing is '+str(x1))
    avg_prob.append(x1)
    x1 = x1/np.average(avg_prob)
    print('')
    print('this is'+str(x1-1)+'% off the average today')
    print('')
    x2 = float(xgb_clf.predict_proba(final_t1)[-1][1])
    print(STOCK1+' XGradient Boosted probability of increasing is '+str(x2))
    avg_probXGB.append(x2)
    x2 = x2/np.average(avg_probXGB)
    print('')
    print('this is'+str(x2-1)+'% off the average today')
    print('')
    x3 = float(gb_clf.predict_proba(final_t1)[-1][1])
    print(STOCK1+' Gradient Boosted probability of increasing is '+str(x3))
    avg_probGB.append(x3)
    x3 = x3/np.average(avg_probGB)
    print('')
    print('this is is'+str(x3-1)+'% off the average today')
    print('')
    if xx>=10:
        if x1 >= 1.05:
            if x2>=1.05:
                if x3 >=1.05:
                    place_order(c, 'buy', 2, STOCK1)
                    bought = str(xx)+STOCK1
                    print('all 3 models probabilities were >5% - '+STOCK1+ ' has been bought')
                else:
                    bought = 0
            else:
                bought = 0
        else:
            bought =0
    else:
        bought = 0
    return bought
                


 
def credit_swish(num_minutes, STOCK1, STOCK2, STOCK3, STOCK4, STOCK5):
    CS1 = stock_setup(STOCK1)
    CS2 = stock_setup(STOCK2)
    CS3 = stock_setup(STOCK3)
    CS4 = stock_setup(STOCK4)
    CS5 = stock_setup(STOCK5)


    for xx in range(num_minutes):
        
        if xx ==0:
            ask_hold = 0
            buy1 = 0
            buy2 = 0
            buy3 = 0
            buy4 = 0
            buy5 = 0
        
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Swish1(c, STOCK1, ask_hold)
        CS1['bid_price'].append(bid_price)
        CS1['bid_quantity'].append(bid_quantity)
        CS1['ask_price'].append(ask_price)
        CS1['ask_quantity'].append(ask_quantity)
        CS1['price'].append(price)
        CS1['pchange'].append(pchange)
        CS1['volume'].append(volume)
        buy1 = printer(STOCK1, CS1,xx, buy1)

        time.sleep(60)

        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Swish1(c, STOCK2, ask_hold)
        CS2['bid_price'].append(bid_price)
        CS2['bid_quantity'].append(bid_quantity)
        CS2['ask_price'].append(ask_price)
        CS2['ask_quantity'].append(ask_quantity)
        CS2['price'].append(price)
        CS2['pchange'].append(pchange)
        CS2['volume'].append(volume)
        buy2 = printer(STOCK2, CS2,xx, buy2)

        
        time.sleep(60)
        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Swish1(c, STOCK3, ask_hold)
        CS3['bid_price'].append(bid_price)
        CS3['bid_quantity'].append(bid_quantity)
        CS3['ask_price'].append(ask_price)
        CS3['ask_quantity'].append(ask_quantity)
        CS3['price'].append(price)
        CS3['pchange'].append(pchange)
        CS3['volume'].append(volume)
        buy3 = printer(STOCK3, CS3,xx, buy3)


        time.sleep(60)

        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Swish1(c, STOCK4, ask_hold)
        CS4['bid_price'].append(bid_price)
        CS4['bid_quantity'].append(bid_quantity)
        CS4['ask_price'].append(ask_price)
        CS4['ask_quantity'].append(ask_quantity)
        CS4['price'].append(price)
        CS4['pchange'].append(pchange)
        CS4['volume'].append(volume)
        buy4 = printer(STOCK4, CS4,xx, buy4)
        

        time.sleep(60)

        bid_price, bid_quantity, ask_price, ask_quantity, price, pchange, volume, ask_hold = Credit_Swish1(c, STOCK5, ask_hold)
        CS5['bid_price'].append(bid_price)
        CS5['bid_quantity'].append(bid_quantity)
        CS5['ask_price'].append(ask_price)
        CS5['ask_quantity'].append(ask_quantity)
        CS5['price'].append(price)
        CS5['pchange'].append(pchange)
        CS5['volume'].append(volume)
        buy5 =  printer(STOCK5, CS5,xx, buy5)

        time.sleep(60)



credit_swish(300,'FUBO', 'PRTY','RIG','CCO','GCI')






