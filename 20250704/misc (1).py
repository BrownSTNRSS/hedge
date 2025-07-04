import pandas as pd
import numpy as np
from scipy.stats import norm
import math

def create_strike_arrays():
    """
    ショートプットとロングプットの行使価格配列を作成
    
    Returns:
    tuple: (strikes_pshort, strikes_plong)
        strikes_pshort: 0.95から1.05まで0.01刻み
        strikes_plong: 0.90から1.00まで0.01刻み
    """
    strikes_pshort = np.arange(95, 106, 1).astype(int)  # 95, 96, ..., 105
    strikes_plong = np.arange(90, 101, 1).astype(int)   # 90, 91, ..., 100
    return strikes_pshort, strikes_plong


def calculate_premium_matrix_with_iv(S, T, r, df_options, date):
    """
    インプライドボラティリティデータフレームを使用してプレミアム差分を計算
    
    Parameters:
    S (float): 現在の原資産価格
    T (float): 満期までの時間（年単位）
    r (float): リスクフリーレート
    df_options (DataFrame): インプライドボラティリティデータフレーム
                           df_options.loc[date, moneyness]でIVが取得可能
    date: データフレームの日付インデックス
    
    Returns:
    tuple: (premium_matrix, strikes_pshort, strikes_plong)
        premium_matrix: 2次元配列 [i, j] = p_short[i] - p_long[j]
        strikes_pshort: ショートプットの行使価格配列
        strikes_plong: ロングプットの行使価格配列
    """
    # 行使価格配列を取得
    strikes_pshort, strikes_plong = create_strike_arrays()
    
    # 結果を格納する2次元配列を初期化
    premium_matrix = np.zeros((len(strikes_pshort), len(strikes_plong)))
    
    # 各組み合わせについてプレミアム差分を計算
    for i, k_short in enumerate(strikes_pshort):
        for j, k_long in enumerate(strikes_plong):
            try:
                # ショートプットのインプライドボラティリティを取得
                sigma_short = df_options.loc[date, k_short]
                
                # ロングプットのインプライドボラティリティを取得
                sigma_long = df_options.loc[date, k_long]
                
                # ショートプットのプレミアム
                put_short = PutOptionPricer(S, S * k_short / 100, T, r, sigma_short)
                p_short = put_short.premium()
                
                # ロングプットのプレミアム
                put_long = PutOptionPricer(S, S * k_long  / 100, T, r, sigma_long)
                p_long = put_long.premium()
                
                # プレミアム差分を格納
                premium_matrix[i, j] = p_short - p_long
                
            except KeyError:
                # 該当するmoneynessのデータが存在しない場合はNaNを設定
                premium_matrix[i, j] = np.nan
                print(f"Warning: IV data not found for date={date}, moneyness_short={k_short:.2f}, moneyness_long={k_long:.2f}")
    
    return premium_matrix, strikes_pshort, strikes_plong


def calculate_delta_matrix_with_iv(S, T, r, df_options, date):
    """
    インプライドボラティリティデータフレームを使用してショートプットのデルタマトリックスを計算
    
    Parameters:
    S (float): 現在の原資産価格
    T (float): 満期までの時間（年単位）
    r (float): リスクフリーレート
    df_options (DataFrame): インプライドボラティリティデータフレーム
    date: データフレームの日付インデックス
    
    Returns:
    tuple: (delta_short_matrix, strikes_pshort, strikes_plong)
        delta_short_matrix: 2次元配列 [i, j] = ショートプット[i]のデルタ
        strikes_pshort: ショートプットの行使価格配列
        strikes_plong: ロングプットの行使価格配列
    """
    # 行使価格配列を取得
    strikes_pshort, strikes_plong = create_strike_arrays()
    
    # 結果を格納する2次元配列を初期化
    delta_short_matrix = np.zeros((len(strikes_pshort), len(strikes_plong)))
    
    # 各組み合わせについてショートプットのデルタを計算
    for i, k_short in enumerate(strikes_pshort):
        for j, k_long in enumerate(strikes_plong):
            try:
                # ショートプットのインプライドボラティリティを取得
                sigma_short = df_options.loc[date, k_short]
                
                # ショートプットのデルタを計算
                put_short = PutOptionPricer(S, S * k_short / 100, T, r, sigma_short)
                delta_short = put_short.delta()
                
                # デルタを格納
                delta_short_matrix[i, j] = delta_short
                
            except KeyError:
                # 該当するmoneynessのデータが存在しない場合はNaNを設定
                delta_short_matrix[i, j] = np.nan
                print(f"Warning: IV data not found for date={date}, moneyness_short={k_short:.2f}")
    
    return delta_short_matrix, strikes_pshort, strikes_plong

def find_optimal_strikes(prob_weighted_return_matrix, strikes_pshort, strikes_plong):
    """
    確率加重リターンマトリックスから最適な行使価格組み合わせを見つける
    
    Parameters:
    prob_weighted_return_matrix: 確率加重リターンマトリックス
    strikes_pshort: ショートプットの行使価格配列
    strikes_plong: ロングプットの行使価格配列
    
    Returns:
    dict: 最適解の情報
    """
    # NaNを除いて最大値を見つける
    valid_mask = ~np.isnan(prob_weighted_return_matrix)
    if not valid_mask.any():
        return {"error": "有効なデータが存在しません"}
    
    # 最大値のインデックスを取得
    max_idx = np.unravel_index(np.nanargmax(prob_weighted_return_matrix), prob_weighted_return_matrix.shape)
    
    optimal_result = {
        "optimal_value": prob_weighted_return_matrix[max_idx],
        "optimal_k_short": strikes_pshort[max_idx[0]],
        "optimal_k_long": strikes_plong[max_idx[1]],
        "optimal_idx": max_idx
    }
    
    return optimal_result


class PutOptionPricer:
    """
    プットオプションのプレミアム、デルタ、ベガを計算するクラス
    Black-Scholesモデルを使用
    """
    
    def __init__(self, S, K, T, r, sigma):
        """
        パラメータを初期化
        
        Parameters:
        S (float): 現在の原資産価格
        K (float): 行使価格
        T (float): 満期までの時間（年単位）
        r (float): リスクフリーレート
        sigma (float): ボラティリティ（年率）
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # d1, d2を計算
        self._calculate_d1_d2()
    
    def _calculate_d1_d2(self):
        """d1とd2を計算"""
        if self.T <= 0:
            raise ValueError("満期までの時間は正の値である必要があります")
        if self.sigma <= 0:
            raise ValueError("ボラティリティは正の値である必要があります")
        if self.S <= 0:
            raise ValueError("原資産価格は正の値である必要があります")
        if self.K <= 0:
            raise ValueError("行使価格は正の値である必要があります")
            
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
    
    def premium(self):
        """
        プットオプションのプレミアムを計算
        
        Returns:
        float: プットオプションの理論価格
        """
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)
        return put_price
    
    def delta(self):
        """
        プットオプションのデルタを計算
        
        Returns:
        float: デルタ（原資産価格に対する感応度）
        """
        put_delta = -norm.cdf(-self.d1)
        return put_delta
    
    def vega(self):
        """
        プットオプションのベガを計算
        
        Returns:
        float: ベガ（ボラティリティに対する感応度）
        """
        vega = self.S * norm.pdf(self.d1) * np.sqrt(self.T)
        return vega
    
    def gamma(self):
        """
        プットオプションのガンマを計算（オプション）
        
        Returns:
        float: ガンマ（デルタの変化率）
        """
        gamma = norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        return gamma
    
    def theta(self):
        """
        プットオプションのシータを計算（オプション）
        
        Returns:
        float: シータ（時間減衰）
        """
        theta = (-self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T)) 
                + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
        return theta
    
    def all_greeks(self):
        """
        すべてのグリークスを一度に計算
        
        Returns:
        dict: プレミアム、デルタ、ベガ、ガンマ、シータを含む辞書
        """
        return {
            'premium': self.premium(),
            'delta': self.delta(),
            'vega': self.vega(),
            'gamma': self.gamma(),
            'theta': self.theta()
        }
    
    def update_parameters(self, S=None, K=None, T=None, r=None, sigma=None):
        """
        パラメータを更新
        
        Parameters:
        S, K, T, r, sigma: 更新したいパラメータ（Noneの場合は変更しない）
        """
        if S is not None:
            self.S = S
        if K is not None:
            self.K = K
        if T is not None:
            self.T = T
        if r is not None:
            self.r = r
        if sigma is not None:
            self.sigma = sigma
            
        # d1, d2を再計算
        self._calculate_d1_d2()



def get_third_fridays(df):
    """
    DataFrameのDatetimeIndex範囲内の第三金曜日のDatetimeIndexを生成
    第三金曜日が休日の場合は翌営業日に調整
    
    Parameters:
    df (DataFrame): DatetimeIndexを持つDataFrame
    
    Returns:
    pd.DatetimeIndex: 第三金曜日（または翌営業日）の日付リスト
    """
    start_date = df.index.min()
    end_date = df.index.max()
    
    third_fridays = []
    current_date = start_date.replace(day=1)
    
    while current_date <= end_date:
        # 月の最初の日
        first_day = current_date
        # 最初の金曜日を見つける（金曜日は4）
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + pd.Timedelta(days=days_until_friday)
        # 第三金曜日は最初の金曜日から14日後
        third_friday = first_friday + pd.Timedelta(days=14)
        
        # 第三金曜日が営業日でない場合は翌営業日を探す
        adjusted_date = get_next_business_day(third_friday, df)
        
        if start_date <= adjusted_date <= end_date:
            third_fridays.append(adjusted_date)
        
        # 次の月へ
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return pd.DatetimeIndex(third_fridays)

def get_next_business_day(date, df):
    """
    指定日がDataFrameに存在しない場合、翌営業日を探す
    
    Parameters:
    date (pd.Timestamp): 調整したい日付
    df (DataFrame): DatetimeIndexを持つDataFrame
    
    Returns:
    pd.Timestamp: 営業日の日付
    """
    # 指定日がDataFrameに存在する場合はそのまま返す
    if date in df.index:
        return date
    
    # 翌営業日を探す（最大10日間）
    for i in range(1, 11):
        next_date = date + pd.Timedelta(days=i)
        if next_date in df.index:
            return next_date
    
    # 10日以内に営業日が見つからない場合は元の日付を返す
    return date