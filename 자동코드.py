import yfinance as yf
import pandas as pd
import pandas_gbq
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay #주식 예측 시평일만 카운트 해주기 위함
from google.oauth2 import service_account
from google.cloud import bigquery

plt.style.use('fivethirtyeight')

key_path = "/home/interimassembly2021/instant-bonfire-426101-f8-f306f7226242_(1).json"


client = bigquery.Client.from_service_account_json(key_path)

#주가 예측& 빅쿼리 연동 함수
def sendtogbq(corp_name):
  current_datetime = datetime.now().strftime("%Y-%m-%d")
  start_year = datetime.now().year - 5          # 현재 년도 기준 몇년 전까지를 할 것인지 : 5년
  start_datetime = datetime.now().replace(year=start_year)
  start_date_str = start_datetime.strftime("%Y-%m-%d")
  corp = yf.download(corp_name, start=start_date_str, end=current_datetime) #입력 날짜 기준으로 주식 데이터 다운로드
  corp['Name'] = str(corp_name)
  corp.columns = ['Open','High','Low','Close','Adj_Close','Volume','Name'] ## Adj_Close값 잘 연동 되도록 열 이름 변경


  corp['ds'] = pd.to_datetime(corp.index, format = '%Y-%m-%d')
  corp['y'] = corp['Adj_Close']           ##조정 마감가를 y에 할당 ++ Adj_Close로 수정// 위에서 열 이름 변경했음
  corp_train = corp[['ds', 'y']][:-251]

  #모델 적합 및 파라미터 설정
  model_prophet = Prophet(changepoint_prior_scale = 0.15, daily_seasonality = True, seasonality_mode='multiplicative', n_changepoints=100, seasonality_prior_scale=0.05)

  #changepoint_prior_scale = 0.15, daily_seasonality = True : 예측력 더 낮으나 상한하한 범위 좁음

  model_prophet.fit(corp)

  #향후 1년간의 time stamp 생성
  fcast_time_with_weekends = 365 #365일 예측
  corp_forecast = model_prophet.make_future_dataframe(periods=fcast_time_with_weekends, freq='D')

  # 주말을 제외한 날짜 범위 생성
  corp_forecast['exclude_weekend'] = corp_forecast['ds'] + BDay(0)
  corp_forecast = corp_forecast[corp_forecast['ds'].dt.weekday < 5]
  corp_forecast = model_prophet.predict(corp_forecast)
  model_prophet.plot(corp_forecast, xlabel = 'Date', ylabel= 'adj price($)')
  plt.show()


  #예측력 테스트
  corp_test = corp[-251:]

  future = corp_test[['ds']]  # 테스트 데이터의 날짜 칼럼을 그대로 사용하여 future 데이터프레임 생성
  forecast = model_prophet.predict(future)

  # 평가 : mae, mse, rmse
  y_pred = forecast['yhat'].values
  y_true = corp_test['y'].values
  mae = mean_absolute_error(y_true, y_pred)
  print('MAE: %.3f' % mae)
  mse = mean_squared_error(y_true, y_pred)
  print('MSE: %.3f' % mse)

  rmse = np.sqrt(mse)
  print('RMSE: %.3f' % rmse)

  plt.plot(y_true, label='Actual')
  plt.plot(y_pred, label='Predicted')
  plt.title('Predictability Test')
  plt.legend()
  plt.show()


  client = bigquery.Client()


  # yfinance 데이터와 예측값 데이터 통합
  corp_forecast = corp_forecast[['ds','yhat_lower', 'yhat_upper', 'yhat']]

  corp_pred = pd.merge(corp,corp_forecast, on=['ds'], how = 'inner')
  corp_pred = corp_pred.sort_values(by='ds')

  table_id = 'strange-terra-415002.practical_project.' + corp_name          #corp_name따라 테이블 만들기
  pandas_gbq.to_gbq(corp_pred, table_id, project_id='strange-terra-415002', if_exists='append') #project에 추가
  #이미 데이터가 있을 경우 추가(실전 프로젝트 할 시에는 중복데이터 처리법을 몰랐었고, 태블로 연동시에 태블로에서는 중복 데이터를 알아서 필터링해줬었던 지라......일단 'append'로 설정했습니다)


  #예측력 측정용 데이터 통합 및 빅쿼리 연동 : 위와 마찬가지로 필요한 부분만 수정
  y_pred_df = pd.DataFrame(y_pred).rename(columns = {0 : "pred"})
  y_true_df = pd.DataFrame(y_true).rename(columns = {0 : "true"})
  combined_df = pd.concat([y_pred_df, y_true_df], axis=1)
  table_id_2 = 'strange-terra-415002.practical_project.' + corp_name + '_predictability' #corp_name따라 예측력 테이블 만들기
  pandas_gbq.to_gbq(combined_df, table_id_2, project_id='strange-terra-415002', if_exists='append') #project에 추가



  #예측 전체 테이블 : 확인용
  return corp_pred




sendtogbq('MSFT')
sendtogbp('AAPL')
