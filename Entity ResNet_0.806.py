import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=== 🚀 启动 [TiDE-Inspired 深度嵌入残差网络] | 使用设备: {device} ===")

# ==========================================
# 1. 气象特征深度处理 (修复了插值索引报错)
# ==========================================
print("[1/5] 正在处理气象特征与节假日日历...")
weather_df = pd.read_csv('fuzhou_weather_2024.csv')
weather_df['日期'] = pd.to_datetime(weather_df['日期'])

# 构建覆盖全年的 15 分钟级时间网格
grid_start, grid_end = pd.to_datetime('2024-01-01 00:00:00'), pd.to_datetime('2024-12-31 23:45:00')
global_weather = pd.DataFrame({'datetime': pd.date_range(grid_start, grid_end, freq='15min')})
global_weather['date'] = global_weather['datetime'].dt.normalize()
global_weather = pd.merge(global_weather, weather_df[['日期', '天气状况', '最高温_摄氏度', '最低温_摄氏度']], left_on='date', right_on='日期', how='left')

# 提取是否下雨
global_weather['is_rain'] = global_weather['天气状况'].fillna('').str.contains('雨').astype(int)

# 温度平滑插值 (14:00最高温, 04:00最低温)
global_weather['temperature'] = np.nan
global_weather.loc[global_weather['datetime'].dt.time == pd.Timestamp('04:00').time(), 'temperature'] = global_weather['最低温_摄氏度']
global_weather.loc[global_weather['datetime'].dt.time == pd.Timestamp('14:00').time(), 'temperature'] = global_weather['最高温_摄氏度']

# 【修复报错的地方】：将 datetime 设为索引进行时间插值，完成后重置索引
global_weather.set_index('datetime', inplace=True)
global_weather['temperature'] = global_weather['temperature'].interpolate(method='time').ffill().bfill()
global_weather.reset_index(inplace=True)

global_weather['temp_diff'] = global_weather['最高温_摄氏度'] - global_weather['最低温_摄氏度']
clean_weather = global_weather[['datetime', 'temperature', 'temp_diff', 'is_rain']].copy()

# 节假日标记 (2024法定节假日)
holidays = ['2024-01-01'] + pd.date_range('2024-02-10','2024-02-17').astype(str).tolist() + \
           pd.date_range('2024-04-04','2024-04-06').astype(str).tolist() + \
           pd.date_range('2024-05-01','2024-05-05').astype(str).tolist() + \
           ['2024-06-10'] + pd.date_range('2024-09-15','2024-09-17').astype(str).tolist() + \
           pd.date_range('2024-10-01','2024-10-07').astype(str).tolist()

# ==========================================
# 2. 训练数据处理与统计锚点提取
# ==========================================
print("[2/5] 正在提取统计锚点与构建特征画像...")
try:
    train_df = pd.read_csv('A榜-充电站充电负荷训练数据.csv', header=1)
except:
    train_df = pd.read_csv('A榜-充电站充电负荷训练数据.csv', header=1, encoding='gbk')

train_df['TIME'] = pd.to_datetime(train_df['TIME'])
train_df['V'] = pd.to_numeric(train_df['V'], errors='coerce')
train_df = train_df.dropna(subset=['V']).copy()

# 将气象数据合并到训练集
train_df = pd.merge(train_df, clean_weather, left_on='TIME', right_on='datetime', how='left')
train_df['month'] = train_df['TIME'].dt.month
train_df['day_of_week'] = train_df['TIME'].dt.dayofweek
train_df['point_in_day'] = train_df['TIME'].dt.hour * 4 + train_df['TIME'].dt.minute // 15
train_df['is_weekend'] = train_df['day_of_week'].isin([5, 6]).astype(int)
train_df['is_holiday'] = train_df['TIME'].dt.normalize().astype(str).isin(holidays).astype(int)

# 核心锚点提取 (以工作日/周末为维度)
anchor_stats = train_df.groupby(['is_weekend', 'point_in_day'])['V'].agg(['median', 'mean', 'std']).reset_index()
anchor_stats.columns = ['is_weekend', 'point_in_day', 'hist_median', 'hist_mean', 'hist_std']

# 近期锚点提取 (捕捉10月份最新趋势)
recent_df = train_df[train_df['TIME'] >= '2024-10-18']
recent_anchor = recent_df.groupby(['is_weekend', 'point_in_day'])['V'].agg('median').reset_index()
recent_anchor.rename(columns={'V': 'recent_14d_median'}, inplace=True)

# 赋予训练集锚点特征
train_full = pd.merge(train_df, anchor_stats, on=['is_weekend', 'point_in_day'], how='left')
train_full = pd.merge(train_full, recent_anchor, on=['is_weekend', 'point_in_day'], how='left')
train_full = train_full.ffill().bfill() # 极少数缺失补全

# ==========================================
# 3. 数据集构建与正则化 (PyTorch 专属)
# ==========================================
print("[3/5] 正在构建 PyTorch 数据集与标准化矩阵...")
cat_cols = ['month', 'day_of_week', 'point_in_day', 'is_weekend', 'is_holiday', 'is_rain']
cont_cols = ['temperature', 'temp_diff', 'hist_median', 'hist_mean', 'hist_std', 'recent_14d_median']

# 类别编码映射到从 0 开始
train_full['month'] = train_full['month'] - 1

# 连续特征标准化
scaler_x = StandardScaler()
X_cont = scaler_x.fit_transform(train_full[cont_cols].values)
X_cat = train_full[cat_cols].values

# 目标值标准化 (极其关键：防止梯度震荡)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(train_full[['V']].values)

class TabularDataset(Dataset):
    def __init__(self, x_cat, x_cont, y=None):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.x_cat)
    def __getitem__(self, idx):
        if self.y is not None: return self.x_cat[idx], self.x_cont[idx], self.y[idx]
        return self.x_cat[idx], self.x_cont[idx]

# 划分最后两周作为验证集，执行早停
val_idx = (train_full['TIME'] >= '2024-10-18').values
train_ds = TabularDataset(X_cat[~val_idx], X_cont[~val_idx], y[~val_idx])
val_ds = TabularDataset(X_cat[val_idx], X_cont[val_idx], y[val_idx])

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

# ==========================================
# 4. SOTA 模型：实体嵌入残差网络 (Entity ResNet)
# ==========================================
class EntityEmbeddingResNet(nn.Module):
    def __init__(self, cat_dims, cont_dim, hidden_dim=256):
        super().__init__()
        # 类别特征 Embedding
        self.embeddings = nn.ModuleList([nn.Embedding(num, dim) for num, dim in cat_dims])
        total_embed_dim = sum([dim for _, dim in cat_dims])
        
        # 连续特征投影
        self.cont_proj = nn.Sequential(
            nn.Linear(cont_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        
        # 融合层
        self.input_proj = nn.Linear(total_embed_dim + 64, hidden_dim)
        
        # 深度残差块 (Deep Residual Blocks) - 提取非线性交叉特征
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(4)
        ])
        
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, cat_x, cont_x):
        emb_outs = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(emb_outs, dim=1)
        x_cont = self.cont_proj(cont_x)
        
        x = F.gelu(self.input_proj(torch.cat([x_cat, x_cont], dim=1)))
        for block in self.blocks:
            x = F.gelu(x + block(x)) # 残差相加
            
        return self.out(x)

# 定义各个类别特征的词汇表大小和嵌入维度
# (月:12->4, 星期:7->4, 时刻:96->16, 周末:2->2, 节假日:2->2, 下雨:2->2)
cat_dims = [(12, 4), (7, 4), (96, 16), (2, 2), (2, 2), (2, 2)]
model = EntityEmbeddingResNet(cat_dims, len(cont_cols)).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=15)
criterion = nn.MSELoss()

# ==========================================
# 5. 模型训练与早停
# ==========================================
print("\n[4/5] 正在训练深度残差网络 (Entity ResNet)...")
EPOCHS = 15
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for cx_cat, cx_cont, cy in train_loader:
        cx_cat, cx_cont, cy = cx_cat.to(device), cx_cont.to(device), cy.to(device)
        optimizer.zero_grad()
        loss = criterion(model(cx_cat, cx_cont), cy)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(cy)
        
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for cx_cat, cx_cont, cy in val_loader:
            cx_cat, cx_cont, cy = cx_cat.to(device), cx_cont.to(device), cy.to(device)
            loss = criterion(model(cx_cat, cx_cont), cy)
            val_loss += loss.item() * len(cy)
            
    train_loss /= len(train_ds)
    val_loss /= len(val_ds)
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_resnet.pth')
        
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ==========================================
# 6. 测试集推演与输出
# ==========================================
print("\n[5/5] 正在加载最优权重，生成 11-12 月终极预测...")
model.load_state_dict(torch.load('best_resnet.pth', map_location=device, weights_only=True))
model.eval()

# 构建测试集特征矩阵
test_dates = pd.date_range('2024-11-01 00:00:00', '2024-12-31 23:45:00', freq='15min')
test_df = pd.DataFrame({'datetime': test_dates})
test_df = pd.merge(test_df, clean_weather, on='datetime', how='left')

test_df['month'] = test_df['datetime'].dt.month - 1
test_df['day_of_week'] = test_df['datetime'].dt.dayofweek
test_df['point_in_day'] = test_df['datetime'].dt.hour * 4 + test_df['datetime'].dt.minute // 15
test_df['is_weekend'] = test_df['day_of_week'].isin([5, 6]).astype(int)
test_df['is_holiday'] = test_df['datetime'].dt.normalize().astype(str).isin(holidays).astype(int)

# 赋予统计锚点
test_df = pd.merge(test_df, anchor_stats, on=['is_weekend', 'point_in_day'], how='left')
test_df = pd.merge(test_df, recent_anchor, on=['is_weekend', 'point_in_day'], how='left')
test_df = test_df.ffill().bfill()

X_test_cont = scaler_x.transform(test_df[cont_cols].values)
X_test_cat = test_df[cat_cols].values
test_ds = TabularDataset(X_test_cat, X_test_cont)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

preds_scaled = []
with torch.no_grad():
    for cx_cat, cx_cont in test_loader:
        cx_cat, cx_cont = cx_cat.to(device), cx_cont.to(device)
        out = model(cx_cat, cx_cont)
        preds_scaled.extend(out.cpu().numpy().flatten())

# 逆标准化并截断负数
predictions = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
predictions = np.clip(predictions, a_min=0, a_max=None)

submission = pd.DataFrame({
    'TIME': test_df['datetime'].dt.strftime('%Y/%m/%d %H:%M').str.replace(' 0', ' ').str.replace('/0', '/'),
    'V': predictions
})
submission.to_csv('Entity_ResNet_Submission.csv', index=False)

print("\n✅ 预测圆满完成！结果已保存为 'Entity_ResNet_Submission.csv'")