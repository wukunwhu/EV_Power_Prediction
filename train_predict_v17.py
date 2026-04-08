import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.cluster import KMeans
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')

base_dir = r"d:\360MoveData\Users\Administrator\Desktop\dataAnalyze"

# ----------- 1. Calendar Logic -----------
def get_day_type(dt_obj):
    d = dt_obj.date()
    # 2024 Holidays/Makeup
    holiday_list = [
        date(2024,1,1),
        date(2024,2,10), date(2024,2,11), date(2024,2,12), date(2024,2,13), date(2024,2,14), date(2024,2,15), date(2024,2,16), date(2024,2,17),
        date(2024,4,4), date(2024,4,5), date(2024,4,6),
        date(2024,5,1), date(2024,5,2), date(2024,5,3), date(2024,5,4), date(2024,5,5),
        date(2024,6,8), date(2024,6,9), date(2024,6,10),
        date(2024,9,15), date(2024,9,16), date(2024,9,17),
        date(2024,10,1), date(2024,10,2), date(2024,10,3), date(2024,10,4), date(2024,10,5), date(2024,10,6), date(2024,10,7)
    ]
    makeup_list = [
        date(2024,2,4), date(2024,2,18), date(2024,4,7), date(2024,4,28), date(2024,5,11), date(2024,9,14), date(2024,9,29), date(2024,10,12)
    ]
    pre_h = [date(2024,1,1)-timedelta(1), date(2024,4,4)-timedelta(1), date(2024,5,1)-timedelta(1), 
             date(2024,6,8)-timedelta(1), date(2024,9,15)-timedelta(1), date(2024,10,1)-timedelta(1),
             date(2024,12,31)]
    
    if d in holiday_list: return 4
    if d in makeup_list: return 3
    if d in pre_h: return 6
    if d.weekday() >= 5: return 2
    return 1

# ----------- 2. Temperature Clustering Analysis -----------
print("V17: Performing Monthly Temperature Clustering...")
df_weather = pd.read_csv(os.path.join(base_dir, "fuzhou_weather_2024.csv"))
df_weather['日期'] = pd.to_datetime(df_weather['日期'])
df_weather['Month'] = df_weather['日期'].dt.month
monthly_temp = df_weather.groupby('Month')['最高温_摄氏度'].mean().reset_index()

# K-Means (K=3) Clustering for 1-12 months
kmeans = KMeans(n_clusters=3, random_state=42)
monthly_temp['Temp_Cluster'] = kmeans.fit_predict(monthly_temp[['最高温_摄氏度']])

# Inspect clusters to order them: Cold=0, Mild=1, Hot=2
cluster_means = monthly_temp.groupby('Temp_Cluster')['最高温_摄氏度'].mean().sort_values()
mapping = {old: new for new, old in enumerate(cluster_means.index)}
monthly_temp['Temp_Cluster'] = monthly_temp['Temp_Cluster'].map(mapping)
print("Clusters Assigned (0:Cold, 1:Mild, 2:Hot):")
print(monthly_temp[['Month', '最高温_摄氏度', 'Temp_Cluster']])

# ----------- 3. KNN Waterline Analysis -----------
train_file = os.path.join(base_dir, "A榜-充电站充电负荷训练数据 copy.csv")
df_train = pd.read_csv(train_file, header=1)
df_train['TIME'] = pd.to_datetime(df_train['TIME'])
df_train['Month'] = df_train['TIME'].dt.month
df_train['Hour'] = df_train['TIME'].dt.hour
df_train['Minute'] = df_train['TIME'].dt.minute
df_train['DAY_TYPE'] = df_train['TIME'].apply(get_day_type)

workdays = df_train[df_train['DAY_TYPE'] == 1]
monthly_v = workdays.groupby('Month')['V'].mean().reset_index().rename(columns={'V': 'Waterline'})
hist_stats = pd.merge(monthly_v, monthly_temp, on='Month').rename(columns={'最高温_摄氏度': 'Avg_Temp'})

def get_knn_waterline(target_temp, k=2):
    t_stats = hist_stats[hist_stats['Month'] <= 10]
    temp_diff = np.abs(t_stats['Avg_Temp'] - target_temp)
    nearest_idx = temp_diff.nsmallest(k).index
    nearest = t_stats.loc[nearest_idx]
    weights = 1.0 / (temp_diff.loc[nearest_idx] + 0.1)
    return (nearest['Waterline'] * weights).sum() / weights.sum()

temp_11 = monthly_temp[monthly_temp['Month']==11]['最高温_摄氏度'].values[0]
temp_12 = monthly_temp[monthly_temp['Month']==12]['最高温_摄氏度'].values[0]
w11 = get_knn_waterline(temp_11)
w12 = get_knn_waterline(temp_12)

# ----------- 4. Global Normalized GBDT Training -----------
print("Normalizing full 1-10 months with Temp_Cluster tags...")
df_train = pd.merge(df_train, monthly_v, on='Month')
df_train['V_Ratio'] = df_train['V'] / df_train['Waterline']
df_train = pd.merge(df_train, monthly_temp[['Month', 'Temp_Cluster']], on='Month')

price_file = os.path.join(base_dir, "附件3 -EV用户充放电电价.csv")
df_price = pd.read_csv(price_file, encoding='gbk')
df_price['Hour'] = df_price['时段'].apply(lambda x: int(x.split(':')[0]))
df_train = pd.merge(df_train, df_price[['Hour', '充电电价(元/kWh)']], on='Hour', how='left')
df_train.rename(columns={'充电电价(元/kWh)': 'Price'}, inplace=True)

features = ['Hour', 'Minute', 'DAY_TYPE', 'Price', 'Temp_Cluster']
cats = ['Hour', 'Minute', 'DAY_TYPE', 'Temp_Cluster']
for c in cats: df_train[c] = df_train[c].astype('category')

print("Training Global Model (Learning cross-regime patterns)...")
model = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=127, random_state=42, verbose=-1)
model.fit(df_train[features], df_train['V_Ratio'])

# ----------- 5. Predict & Reconstruct -----------
print("Generating V17 climatized-global forecast...")
test_times = pd.date_range('2024-11-01', '2024-12-31 23:45:00', freq='15min')
df_test = pd.DataFrame({'TIME': test_times})
df_test['Month'] = df_test['TIME'].dt.month
df_test['Hour'] = df_test['TIME'].dt.hour
df_test['Minute'] = df_test['TIME'].dt.minute
df_test['DAY_TYPE'] = df_test['TIME'].apply(get_day_type)
df_test = pd.merge(df_test, df_price[['Hour', '充电电价(元/kWh)']], on='Hour', how='left')
df_test.rename(columns={'充电电价(元/kWh)': 'Price'}, inplace=True)
df_test = pd.merge(df_test, monthly_temp[['Month', 'Temp_Cluster']], on='Month')

for c in cats: df_test[c] = df_test[c].astype('category')

df_test['V'] = model.predict(df_test[features])
df_test.loc[df_test['Month']==11, 'V'] *= w11
df_test.loc[df_test['Month']==12, 'V'] *= w12

# ----------- 6. Export -----------
def format_time(ts):
    return f"{ts.year}/{ts.month}/{ts.day} {ts.hour}:{ts.minute:02d}"

sub_df = pd.DataFrame({
    'TIME': df_test['TIME'].apply(format_time),
    'V': np.round(df_test['V'], 15)
})
sub_df.to_csv(os.path.join(base_dir, 'submission.csv'), index=False)
print("V17 Global Clustered Model Complete.")
