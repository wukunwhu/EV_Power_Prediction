import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== 🚀 启动 [中位数锚点 + 动态气象] 终极版 ===")

# ==========================================
# 1. 气象特征处理 (你的加分神装)
# ==========================================
print("\n[1/4] 正在处理 15 分钟级气象特征...")
weather_df = pd.read_csv('new_plan/fuzhou_weather_2024.csv')
weather_df['日期'] = pd.to_datetime(weather_df['日期'])

grid_start, grid_end = pd.to_datetime('2024-01-01 00:00:00'), pd.to_datetime('2024-12-31 23:45:00')
global_weather = pd.DataFrame({'datetime': pd.date_range(grid_start, grid_end, freq='15min')})
global_weather['date'] = global_weather['datetime'].dt.normalize()
global_weather = pd.merge(global_weather, weather_df[['日期', '最高温_摄氏度', '最低温_摄氏度']], left_on='date', right_on='日期', how='left')

global_weather['temperature'] = np.nan
global_weather.loc[global_weather['datetime'].dt.time == pd.Timestamp('04:00').time(), 'temperature'] = global_weather['最低温_摄氏度']
global_weather.loc[global_weather['datetime'].dt.time == pd.Timestamp('14:00').time(), 'temperature'] = global_weather['最高温_摄氏度']
global_weather['temperature'] = global_weather['temperature'].interpolate(method='polynomial', order=3).bfill().ffill()
global_weather['temp_lag_1h'] = global_weather['temperature'].shift(4).bfill()
clean_weather = global_weather[['datetime', 'temperature', 'temp_lag_1h']]

# ==========================================
# 2. 准备训练集 & 构建【中位数锚点】
# ==========================================
print("[2/4] 正在构建画像与中位数锚点特征...")
train_full = pd.read_csv('new_plan/A榜-充电站充电负荷训练数据.csv', header=1) 
train_full['datetime'] = pd.to_datetime(train_full['TIME'])
train_full = pd.merge(train_full, clean_weather, on='datetime', how='left')

train_full['day_of_week'] = train_full['datetime'].dt.dayofweek
train_full['hour'] = train_full['datetime'].dt.hour
train_full['minute'] = train_full['datetime'].dt.minute
train_full['point_in_day'] = train_full['hour'] * 4 + train_full['minute'] // 15 

# 💡 核心魔法：构建画像！
# 按照 "星期几" 和 "一天中的第几个点"，计算历史所有样本的【功率中位数】
# 比如：计算所有星期一，早上 08:15 的功率中位数
profile_median = train_full.groupby(['day_of_week', 'point_in_day'])['V'].median().reset_index()
profile_median.rename(columns={'V': 'median_anchor_V'}, inplace=True)

# 将中位数锚点作为一个强力特征，合并回训练集
train_full = pd.merge(train_full, profile_median, on=['day_of_week', 'point_in_day'], how='left')

# ==========================================
# 3. 模型训练 (基于锚点去拟合真实波动)
# ==========================================
print("\n[3/4] 🤖 正在训练 LightGBM 模型...")
# 把你的画像锚点 (median_anchor_V) 作为最重要的特征喂进去
features = ['median_anchor_V', 'point_in_day', 'day_of_week', 'temperature', 'temp_lag_1h']
cat_features = ['point_in_day', 'day_of_week']

lgb_train = lgb.Dataset(train_full[features], label=train_full['V'], categorical_feature=cat_features)

params = {
    'objective': 'regression', 
    'metric': 'rmse', 
    'learning_rate': 0.01, 
    'num_leaves': 31,     # 有了强锚点，不需要太复杂的树结构，防止过拟合
    'max_depth': 6,
    'verbose': -1,
    'random_state': 42
}
model = lgb.train(params, lgb_train, num_boost_round=800)

# ==========================================
# 4. 测试集预测与导出
# ==========================================
print("\n[4/4] 正在将画像映射到 11-12 月并预测...")
test_dates = pd.date_range('2024-11-01 00:00:00', '2024-12-31 23:45:00', freq='15min')
test_df = pd.DataFrame({'datetime': test_dates})
test_df = pd.merge(test_df, clean_weather, on='datetime', how='left')

test_df['day_of_week'] = test_df['datetime'].dt.dayofweek
test_df['hour'] = test_df['datetime'].dt.hour
test_df['minute'] = test_df['datetime'].dt.minute
test_df['point_in_day'] = test_df['hour'] * 4 + test_df['minute'] // 15

# 将训练集里算好的【中位数锚点】无缝映射到测试集的每一天！
test_df = pd.merge(test_df, profile_median, on=['day_of_week', 'point_in_day'], how='left')

# 预测最终功率 V
test_df['V'] = model.predict(test_df[features])

# 格式化导出
test_df['TIME'] = test_df['datetime'].apply(lambda x: f"{x.year}/{x.month}/{x.day} {x.hour}:{x.minute:02d}")
submission = test_df[['TIME', 'V']].copy()
submission.to_csv('submission_lgb_median_weather.csv', index=False, encoding='utf-8')

print("\n" + "="*50)
print("🏆 画像构建完成！文件: submission_lgb_median_weather.csv")
print("拥有 0.779 的定海神针 + 气象敏锐度，这版必出好成绩！")
print("="*50)