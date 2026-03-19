import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re  # <--- 新增：导入正则表达式库

def scrape_fuzhou_weather(year, start_month, end_month):
    all_data = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for month in range(start_month, end_month + 1):
        month_str = f"{month:02d}"
        url = f"http://www.tianqihoubao.com/lishi/fuzhou/month/{year}{month_str}.html"
        print(f"正在爬取 {year}年{month}月 的数据...")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            # 建议：可以让 requests 自动识别编码，或者遇到乱码时容错处理
            # 删掉 response.encoding = 'gbk'，或者改为：
            response.encoding = response.apparent_encoding 
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            if not table:
                continue
                
            rows = table.find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 4:
                    date = cols[0].text.strip().replace('\r\n', '').replace(' ', '')
                    weather = cols[1].text.strip().replace('\r\n', '').replace(' ', '')
                    temp = cols[2].text.strip().replace('\r\n', '').replace(' ', '')
                    wind = cols[3].text.strip().replace('\r\n', '').replace(' ', '')
                    
                    # === 修改部分开始 ===
                    # 使用正则匹配所有数字（包含可能存在的负号，如 -5）
                    # 这样不论后面跟着 '℃' 还是 '鈩' 或者是 '度'，都能无视它们
                    temps = re.findall(r'-?\d+', temp)
                    
                    if len(temps) >= 2:
                        high_temp = int(temps[0])
                        low_temp = int(temps[1])
                    else:
                        high_temp, low_temp = None, None
                    # === 修改部分结束 ===
                        
                    all_data.append({
                        '日期': date,
                        '天气状况': weather,
                        '最高温_摄氏度': high_temp,
                        '最低温_摄氏度': low_temp,
                        '风力风向': wind
                    })
            time.sleep(random.uniform(1.5, 3.0))
        except Exception as e:
            print(f"  [错误] 爬取 {year}年{month}月 时发生异常: {e}")

    df = pd.DataFrame(all_data)
    # 清洗日期格式
    df['日期'] = df['日期'].str.replace('年', '-').str.replace('月', '-').str.replace('日', '')
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce') # 加入 errors='coerce' 防止个别脏数据阻断程序
    
    return df

# 重新运行你的爬虫
target_year = 2024  # 请确认你需要爬取的年份是否包含 2025
weather_df = scrape_fuzhou_weather(target_year, 1, 12)
print(weather_df.head())

# 保存为 CSV 文件
weather_df.to_csv('new_plan/fuzhou_weather_2024.csv', index=False, encoding='utf-8-sig')
print("数据已成功保存至 fuzhou_weather_2024.csv")