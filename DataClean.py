        # Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load the data
df = pd.read_csv("ekpc_usage.csv")  # Make sure the file is in the same folder
df.head()

# Step 3: Parse timestamp
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
df['Month'] = df['Datetime'].dt.month

# Step 4: Check for missing values
print("Missing values:\n", df.isnull().sum())

# Step 5: Visualize daily usage pattern
plt.figure(figsize=(12,5))
df.groupby('Hour')['EKPC_MW'].mean().plot(kind='bar', color='skyblue')
plt.title("Average EKPC Usage by Hour of Day")
plt.ylabel("Usage (MW)")
plt.xlabel("Hour")
plt.show()

# Step 6: Visualize weekly pattern
plt.figure(figsize=(12,5))
df.groupby('DayOfWeek')['EKPC_MW'].mean().plot(kind='bar', color='orange')
plt.title("Average EKPC Usage by Day of Week")
plt.ylabel("Usage (MW)")
plt.xlabel("Day of Week (0=Monday)")
plt.show()

# Step 7: Visualize monthly/seasonal pattern
plt.figure(figsize=(12,5))
df.groupby('Month')['EKPC_MW'].mean().plot(kind='bar', color='green')
plt.title("Average EKPC Usage by Month")
plt.ylabel("Usage (MW)")
plt.xlabel("Month")
plt.show()
