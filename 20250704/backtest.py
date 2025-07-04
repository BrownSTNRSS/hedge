
import pandas as pd
from misc import *
import datetime
import matplotlib.pyplot as plt


# データ作成（サンプルデータ）
df = pd.read_csv('SPY_VIX.csv',index_col=0)
df.index = pd.to_datetime(df.index)
df = df.rename(columns = {'SPY_Close':'SPY', "VIX_Close":'VIX'})
df['VIX'] = df['VIX'] / 100

df_options = pd.DataFrame(index=df.index)
all_moneyness = np.linspace(90, 105, 16).astype(int)
for mny in all_moneyness:
    df_options[mny] = df['VIX'] * ((1.05 - mny/100)**2 + 1) 
df["rf"] = 0.01

# 第三金曜日（ないなら翌営業日）
df_backtest = df.loc[get_third_fridays(df)]
df_backtest['Return'] = df_backtest['SPY'].pct_change()
df_backtest.dropna(inplace=True)


# バックテスト
wealth_list = list()
result_list = list()

for date in df_backtest.index:
    # 市場の状態
    idx_date = df_options.index.get_loc(date)
    last_date = df_options.index[idx_date-1]
    S = df_backtest.loc[date, 'SPY']
    r = df_backtest.loc[date,'rf']
    T = 1/12
    
    # ペイオフ計算
    if date != df_backtest.index[0]:
        payoff_all = -np.maximum(S_last * K_short / 100 - S, 0) + np.maximum(S_last * K_long / 100 - S, 0) + payoff_premium# payoff premiumは前月時点で確定
        wealth_list.append(payoff_all / S_last)
    else:
        wealth_list.append(0)
        

    # 意思決定
    premium_matrix, strikes_pshort, strikes_plong = calculate_premium_matrix_with_iv(S, T, r, df_options, last_date) # PUT Short Premium - PUT Long Premium
    delta_short_matrix, strikes_pshort, strikes_plong = calculate_delta_matrix_with_iv(S, T, r, df_options, last_date)
    prob_pshort_matrix = 1 - np.abs(delta_short_matrix)
    weighted_return_matrix = (premium_matrix / S) * prob_pshort_matrix
    optimal_result = find_optimal_strikes(weighted_return_matrix, strikes_pshort, strikes_plong)
    result_list.append(optimal_result)

    # 価格計算
    K_short = optimal_result["optimal_k_short"]
    K_long = optimal_result["optimal_k_long"]
    pshort = PutOptionPricer(S, S * K_short / 100, T, r, df_options.loc[date, K_short]) 
    plong = PutOptionPricer(S, S * K_long / 100, T, r, df_options.loc[date, K_long])
    payoff_premium = pshort.premium() - plong.premium()
    S_last = S


# 結果集計
df_result = pd.DataFrame(index = df_backtest.index)
df_result['SPY'] = df_backtest['Return'].values
df_result.iloc[0,0] = 0
df_result['strategy'] = wealth_list