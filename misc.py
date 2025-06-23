import yfinance as yf
import pandas as pd

# SPYの日次データを取得
spy = yf.Ticker("SPY")
spy_data = spy.history(period="max", interval="1d")
spy_close = spy_data[['Close']].copy()
spy_close.columns = ['SPY_Close']

print(f"SPYデータ取得完了: {len(spy_close)}日分")
print(f"期間: {spy_close.index[0].date()} から {spy_close.index[-1].date()}")

# VIXの日次データを取得
vix = yf.Ticker("^VIX")
vix_data = vix.history(period="max", interval="1d")
vix_close = vix_data[['Close']].copy()
vix_close.columns = ['VIX_Close']

print(f"VIXデータ取得完了: {len(vix_close)}日分")
print(f"期間: {vix_close.index[0].date()} から {vix_close.index[-1].date()}")
spy_close.index = spy_close.index.date
vix_close.index = vix_close.index.date
spy_close.index = pd.to_datetime(spy_close.index)
vix_close.index = pd.to_datetime(vix_close.index)


# SPYとVIXのデータを結合
data = pd.merge(spy_close, vix_close, left_index=True, right_index=True, how='inner')

print(f"\n結合後のデータ: {len(data)}日分")
print(f"期間: {data.index[0].date()} から {data.index[-1].date()}")

# 基本統計量を表示
print("\n基本統計量:")
print(data.describe())

# 最初の5行を表示
print("\n最初の5行:")
print(data.head())



# 日次リターンとボラティリティを計算
data['SPY_Return'] = data['SPY_Close'].pct_change()
data['SPY_Monthly_Return'] = data['SPY_Return'].rolling(window=21).mean() * 252
data['Rolling_Vol'] = data['SPY_Return'].rolling(window=21).std() * np.sqrt(252)  # 20日ローリング、年率換算

# 日次データフレームを作成（NaN削除）
daily_data = pd.DataFrame({
    'Return': data['SPY_Return'],
    'Rolling_Return': data['SPY_Monthly_Return'],
    'Volatility': data['Rolling_Vol'],
    'VIX': data['VIX_Close'],
    'SPY_Close': data['SPY_Close']
})
daily_data = daily_data.dropna()

# GMMの入力特徴量を準備（日次リターンとボラティリティ）
features = daily_data[["Rolling_Return", 'VIX', 'Volatility']].values

# ガウス混合モデルでレジームを推定（3レジーム）
gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
gmm.fit(features)

# レジーム確率とラベルを取得
regime_probs = gmm.predict_proba(features)
regime_labels = gmm.predict(features)

# スムージング
regime_probs = pd.DataFrame(regime_probs).rolling(3).mean().values
regime_labels = np.argmax(regime_probs,axis=1)


# 各レジームの特徴を計算してレジーム名を割り当て
regime_stats = []
for i in range(3):
    mask = regime_labels == i
    avg_return = daily_data.loc[mask, 'Return'].mean()
    avg_vol = daily_data.loc[mask, 'Volatility'].mean()
    avg_vix = daily_data.loc[mask, 'VIX'].mean()
    count = mask.sum()
    regime_stats.append({
        'regime_id': i,
        'avg_return': avg_return,
        'avg_vol': avg_vol,
        'avg_vix': avg_vix,
        'count': count
    })

# リターンでソートしてBear, Neutral, Bullを割り当て
regime_stats = sorted(regime_stats, key=lambda x: x['avg_return'])
regime_mapping = {regime_stats[0]['regime_id']: 'Bear',
                 regime_stats[1]['regime_id']: 'Neutral', 
                 regime_stats[2]['regime_id']: 'Bull'}

# レジーム名を割り当て
daily_data['Regime'] = [regime_mapping[label] for label in regime_labels]
daily_data['Bear_Prob'] = regime_probs[:, regime_stats[0]['regime_id']]
daily_data['Neutral_Prob'] = regime_probs[:, regime_stats[1]['regime_id']]
daily_data['Bull_Prob'] = regime_probs[:, regime_stats[2]['regime_id']]
daily_data = daily_data.dropna()

# レジーム統計を表示
print("レジーム統計（日次）:")
for i, stats in enumerate(regime_stats):
    regime_name = ['Bear', 'Neutral', 'Bull'][i]
    print(f"{regime_name}: 平均リターン={stats['avg_return']*252:.1%}（年率）, "
          f"平均ボラティリティ={stats['avg_vol']:.1%}, "
          f"平均VIX={stats['avg_vix']:.1f}, "
          f"日数={stats['count']:,}日")

# レジーム可視化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 散布図でレジーム分類を表示（サンプリングして見やすくする）
sample_size = min(5000, len(daily_data))
sample_idx = np.random.choice(len(daily_data), sample_size, replace=False)
sample_data = daily_data.iloc[sample_idx]

colors = {'Bear': 'red', 'Neutral': 'gold', 'Bull': 'green'}
for regime in ['Bear', 'Neutral', 'Bull']:
    mask = sample_data['Regime'] == regime
    ax1.scatter(sample_data.loc[mask, 'Return'] * 100, 
               sample_data.loc[mask, 'Volatility'] * 100,
               c=colors[regime], label=regime, alpha=0.5, s=10)
ax1.set_xlabel('日次リターン (%)')
ax1.set_ylabel('ボラティリティ（年率 %）')
ax1.set_title('レジーム分類（日次リターン vs ボラティリティ）')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-5, 5)  # 外れ値を除外して見やすくする

# レジーム確率の積み上げグラフ（月次で集約して見やすくする）
monthly_probs = daily_data[['Bear_Prob', 'Neutral_Prob', 'Bull_Prob']].resample('M').mean()

ax2.fill_between(monthly_probs.index, 0, monthly_probs['Bear_Prob'], 
                 color='red', alpha=0.7, label='Bear')
ax2.fill_between(monthly_probs.index, monthly_probs['Bear_Prob'], 
                 monthly_probs['Bear_Prob'] + monthly_probs['Neutral_Prob'], 
                 color='gold', alpha=0.7, label='Neutral')
ax2.fill_between(monthly_probs.index, 
                 monthly_probs['Bear_Prob'] + monthly_probs['Neutral_Prob'], 
                 1.0, color='green', alpha=0.7, label='Bull')
ax2.set_ylabel('レジーム確率')
ax2.set_ylim(0, 1)
ax2.set_title('レジーム確率の推移（月次平均、積み上げグラフ）')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 各レジームの期間を可視化
fig, ax = plt.subplots(figsize=(14, 4))
regime_colors = daily_data['Regime'].map({'Bear': 0, 'Neutral': 1, 'Bull': 2})
ax.scatter(daily_data.index, daily_data['SPY_Close'], 
          c=regime_colors, cmap='RdYlGn', s=1, alpha=0.5)
ax.set_ylabel('SPY価格')
ax.set_title('S&P500価格とレジーム（赤:Bear、黄:Neutral、緑:Bull）')
ax.grid(True, alpha=0.3)
plt.show()

from scipy.stats import norm

# プットオプション価格計算（Black-Scholesモデル）
def calculate_put_price(S, K, r, sigma, T):
    """
    S: 現在の株価
    K: 行使価格
    r: リスクフリーレート
    sigma: ボラティリティ（年率）
    T: 満期までの時間（年）
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return put_price



# 面倒なので月末を取っている
monthly_data = daily_data.resample('M').last()
monthly_data["SPY_Close"] = monthly_data["SPY_Close"].pct_change()
monthly_data.dropna(inplace=True)




# パラメータ設定
r = 0.02  # リスクフリーレート（年率2%）
T = 1/12  # 1ヶ月満期
moneyness_levels = [0.95, 0.97, 0.99, 1.00]  # マネネスレベル

# 各月・各マネネスでのプットオプション価格を計算
for moneyness in moneyness_levels:
    put_prices = []
    
    for idx, row in monthly_data.iterrows():
        S = data['SPY_Close'].loc[:idx].iloc[-1]  # 月末のSPY価格
        K = S * moneyness  # 行使価格
        sigma = row['VIX'] / 100
        
        put_price = calculate_put_price(S, K, r, sigma, T)
        put_prices.append(put_price / S)  # SPY価格に対する比率として保存
    
    monthly_data[f'Put_Price_{moneyness}'] = put_prices

# プットオプション価格の統計を表示
print("\nプットオプション価格（SPY価格に対する比率）:")
for moneyness in moneyness_levels:
    col = f'Put_Price_{moneyness}'
    mean_price = monthly_data[col].mean()
    print(f"マネネス{moneyness}: 平均{mean_price:.3f} ({mean_price*100:.1f}%)")

# プットオプション価格の可視化
fig, ax = plt.subplots(figsize=(12, 6))
for moneyness in moneyness_levels:
    monthly_data[f'Put_Price_{moneyness}'].plot(ax=ax, label=f'マネネス{moneyness}')
ax.set_ylabel('プット価格 / SPY価格')
ax.set_title('プットオプション価格の推移（SPY価格に対する比率）')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()