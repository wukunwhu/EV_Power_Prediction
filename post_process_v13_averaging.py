import pandas as pd
import numpy as np

# 1. Loading the v13 ensemble results
print("Loading submit_result_v13_ensemble.csv...")
df = pd.read_csv('submit_result_v37_auto_avg.csv')
df['TIME'] = pd.to_datetime(df['TIME'])

# 2. Extracting basic features for grouping
df['month'] = df['TIME'].dt.month
df['day'] = df['TIME'].dt.day
df['hour'] = df['TIME'].dt.hour
df['minute'] = df['TIME'].dt.minute
df['dayofweek'] = df['TIME'].dt.dayofweek

# 3. Defining Workday/Weekend Logic (following v13/v16 standards)
workdays_override = {
    '2024-02-04', '2024-02-18', '2024-04-07', '2024-04-28', '2024-05-11', '2024-09-14', '2024-09-29', '2024-10-12',
}
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['DATE_STR'] = df['TIME'].dt.date.astype(str)
df.loc[df['DATE_STR'].isin(workdays_override), 'is_weekend'] = 0

# 4. For each (month, is_weekend, hour, minute), calculate the average profile
print("Calculating average daily profiles for each month and day-type...")
# Group by Month, Weekend-Status, and Time-of-Day
profile_means = df.groupby(['month', 'is_weekend', 'hour', 'minute'])['V'].mean().reset_index()
profile_means.rename(columns={'V': 'V_avg_profile'}, inplace=True)

# 5. Merge back and replace the V values
df = pd.merge(df, profile_means, on=['month', 'is_weekend', 'hour', 'minute'], how='left')
df['V'] = df['V_avg_profile']

# 6. Cleaning up and saving
output_file = 'submit_result_v37_auto_avg_avg.csv'
df_final = df[['TIME', 'V']]
df_final.to_csv(output_file, index=False)

print(f"Done! Created '{output_file}' with average-profile values.")
print("Statistics:")
for mon in [11, 12]:
    for weekend in [0, 1]:
        type_str = "Weekend" if weekend == 1 else "Workday"
        mean_val = df[(df['month']==mon) & (df['is_weekend']==weekend)]['V'].mean()
        print(f"  Month {mon} {type_str} Global Mean: {mean_val:.4f}")
