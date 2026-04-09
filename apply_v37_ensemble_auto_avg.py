import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

# 1. Loading data
print("1. Loading datasets...")
try:
    df_train_raw = pd.read_csv('A榜-充电站充电负荷训练数据.csv', header=1, encoding='utf-8')
except Exception:
    df_train_raw = pd.read_csv('A榜-充电站充电负荷训练数据.csv', header=1, encoding='gbk')

df_weather = pd.read_csv('fuzhou_weather_2024.csv')
df_sub_raw = pd.read_csv('submit_example.csv')

# 2. Preprocessing
print("2. Preprocessing training data & weather...")
df_train = df_train_raw.groupby('TIME', as_index=False)['V'].sum()
df_train['TIME'] = pd.to_datetime(df_train['TIME'])

df_weather.columns = ['DATE', 'WEATHER', 'HIGH_TEMP', 'LOW_TEMP', 'WIND']
df_weather['DATE'] = pd.to_datetime(df_weather['DATE']).dt.date
for col in ['HIGH_TEMP', 'LOW_TEMP']:
    df_weather[col] = pd.to_numeric(df_weather[col].astype(str).str.replace('℃',''), errors='coerce')
df_weather['avg_temp'] = (df_weather['HIGH_TEMP'] + df_weather['LOW_TEMP']) / 2
df_weather['temp_diff'] = df_weather['HIGH_TEMP'] - df_weather['LOW_TEMP']

# Temperature Anomaly
df_weather['month_raw'] = pd.to_datetime(df_weather['DATE']).dt.month
monthly_avg_map = df_weather.groupby('month_raw')['avg_temp'].mean().to_dict()
df_weather['monthly_avg_temp'] = df_weather['month_raw'].map(monthly_avg_map)
df_weather['temp_anomaly'] = df_weather['avg_temp'] - df_weather['monthly_avg_temp']
df_weather['delta_temp'] = df_weather['avg_temp'].diff().fillna(0)

# Holiday list (Standard)
workdays_override = {'2024-02-04', '2024-02-18', '2024-04-07', '2024-04-28', '2024-05-11', '2024-09-14', '2024-09-29', '2024-10-12'}
golden_week = {'2024-02-10','2024-02-11','2024-02-12','2024-02-13','2024-02-14','2024-02-15','2024-02-16','2024-02-17','2024-10-01','2024-10-02','2024-10-03','2024-10-04','2024-10-05','2024-10-06','2024-10-07'}
short_holiday = {'2024-01-01', '2024-04-04', '2024-04-05', '2024-04-06', '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05', '2024-06-08', '2024-06-09', '2024-06-10', '2024-09-15', '2024-09-16', '2024-09-17'}
all_holidays = golden_week | short_holiday

# 3. Merging and Dynamic Month Alignment
print("3. Merging and Dynamic Month Alignment...")
df_sub = df_sub_raw.copy()
df_sub['TIME'] = pd.to_datetime(df_sub['TIME'])

df_all = pd.concat([
    pd.DataFrame({'TIME': df_train['TIME'], 'V': df_train['V'], 'is_train': 1}),
    pd.DataFrame({'TIME': df_sub['TIME'], 'V': np.nan, 'is_train': 0})
], ignore_index=True)

df_all['DATE'] = df_all['TIME'].dt.date
df_all = pd.merge(df_all, df_weather[['DATE', 'avg_temp', 'temp_diff', 'temp_anomaly', 'delta_temp']], on='DATE', how='left')

# Dynamic Alignment Logic
train_months = sorted(df_all[df_all['is_train']==1]['TIME'].dt.month.unique())
test_months = sorted(df_all[df_all['is_train']==0]['TIME'].dt.month.unique())
target_months = [m for m in test_months if m not in train_months]

df_all['month'] = df_all['TIME'].dt.month

if target_months:
    print(f"  -> Identifying most similar months for {target_months}...")
    monthly_weather = df_weather.groupby('month_raw').agg({
        'HIGH_TEMP': 'mean',
        'LOW_TEMP': 'mean',
        'avg_temp': 'mean'
    })
    
    mapping_dict = {}
    for tm in target_months:
        target_stats = monthly_weather.loc[tm]
        best_m, min_dist = -1, float('inf')
        for rm in train_months:
            ref_stats = monthly_weather.loc[rm]
            dist = np.sqrt(
                (target_stats['HIGH_TEMP'] - ref_stats['HIGH_TEMP'])**2 +
                (target_stats['LOW_TEMP'] - ref_stats['LOW_TEMP'])**2 +
                (target_stats['avg_temp'] - ref_stats['avg_temp'])**2
            )
            if dist < min_dist:
                min_dist, best_m = dist, rm
        mapping_dict[tm] = best_m
        print(f"  -> Auto Alignment Success: {tm} aligned to {best_m}")
    
    for tm, rm in mapping_dict.items():
        df_all.loc[df_all['TIME'].dt.month == tm, 'month'] = rm

df_all['dayofweek'] = df_all['TIME'].dt.dayofweek
df_all['hour'] = df_all['TIME'].dt.hour
df_all['minute'] = df_all['TIME'].dt.minute

df_all['is_weekend'] = df_all['dayofweek'].isin([5, 6]).astype(int)
df_all.loc[df_all['DATE'].astype(str).isin(workdays_override), 'is_weekend'] = 0
df_all['is_holiday'] = df_all['DATE'].astype(str).isin(all_holidays).astype(int)

# Target Encoding
print("4. Target Encoding...")
te_mask = (df_all['is_train'] == 1)
te_map = df_all[te_mask].groupby(['is_weekend', 'hour', 'minute'])['V'].mean().reset_index()
te_map.rename(columns={'V': 'hist_mean_V'}, inplace=True)
df_all = pd.merge(df_all, te_map, on=['is_weekend', 'hour', 'minute'], how='left')
df_all['hist_mean_V'] = df_all['hist_mean_V'].fillna(df_all[df_all['is_train']==1]['V'].mean())

# Interactions
df_all['is_deep_valley'] = df_all['hour'].isin([2, 3, 4, 5, 6]).astype(int)
df_all['is_eve_peak'] = df_all['hour'].isin([19, 20, 21]).astype(int)

# 5. Training Ensemble
print("5. Training Ensemble Models...")
train_mask = (df_all['is_train'] == 1)
test_mask = (df_all['is_train'] == 0)

feats = ['month', 'dayofweek', 'hour', 'minute', 'is_weekend', 'is_holiday', 
         'avg_temp', 'temp_diff', 'temp_anomaly', 'delta_temp', 
         'hist_mean_V', 'is_deep_valley', 'is_eve_peak']

X_train, y_train = df_all[train_mask][feats], df_all[train_mask]['V']
X_test = df_all[test_mask][feats]

m_lgb = lgb.LGBMRegressor(n_estimators=2500, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
m_lgb.fit(X_train, y_train)
p_lgb = m_lgb.predict(X_test)

m_xgb = xgb.XGBRegressor(n_estimators=2500, learning_rate=0.03, max_depth=6, random_state=42)
m_xgb.fit(X_train, y_train)
p_xgb = m_xgb.predict(X_test)

m_cat = cb.CatBoostRegressor(iterations=2500, learning_rate=0.03, depth=6, random_seed=42, verbose=False)
m_cat.fit(X_train, y_train)
p_cat = m_cat.predict(X_test)

# 6. Ensemble and INTEGRATED AVERAGING
print("6. Ensemble Blending & Calendar-Type Averaging...")
test_df = df_all[test_mask].copy()
test_df['V_pred'] = (p_lgb + p_xgb + p_cat) / 3.0
test_df['orig_month'] = test_df['TIME'].dt.month

# Group by Original Month, Weekend Status, and Time of Day
test_df['V_final'] = test_df.groupby(['orig_month', 'is_weekend', 'hour', 'minute'])['V_pred'].transform('mean')

df_sub_raw['V'] = np.clip(test_df['V_final'].values, 0, None)

# 7. Final Smoothing
print("7. Applying SG Filtering...")
df_sub_raw['V'] = savgol_filter(df_sub_raw['V'], window_length=7, polyorder=2)

output_file = 'submit_result_v37_auto_avg.csv'
df_sub_raw.to_csv(output_file, index=False)
print(f"Success! Final integrated result saved to {output_file}")
