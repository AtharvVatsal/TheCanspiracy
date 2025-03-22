import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import squarify
import matplotlib.dates as mdates
import datetime

# Load the CSV file
df = pd.read_csv(r"A:\Coding\Can Detection\yolo_training_results.csv")

# Clean and preprocess the data
print("Column Names:", df.columns)
df.columns = df.columns.str.strip()
columns_to_convert = ["Instances", "Precision (P)", "Recall (R)", "mAP50", "mAP50-95"]
for col in columns_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Column '{col}' not found in CSV")
df = df[df["Class"] != "all"]

sns.set_theme(style="whitegrid")

# Sort data by mAP50
df_sorted = df.sort_values(by="mAP50", ascending=False)

# 1. Bar Chart (Precision per class)
plt.figure(figsize=(10, 6))
sns.barplot(data=df_sorted, x="Precision (P)", y="Class", palette="Blues_r")
plt.xlabel("Precision (P)")
plt.ylabel("Class")
plt.title("Precision per Class")
plt.show()

# 2. Scatter Plot (Precision vs Recall)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Precision (P)", y="Recall (R)", hue="Class", size="Instances", sizes=(20, 200))
plt.xlabel("Precision (P)")
plt.ylabel("Recall (R)")
plt.title("Precision vs Recall")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
plt.show()

# 3. Line Chart (mAP50 over classes)
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_sorted, x="Class", y="mAP50", marker="o", color="r")
plt.xticks(rotation=45)
plt.xlabel("Class")
plt.ylabel("mAP50")
plt.title("mAP50 per Class (Line Chart)")
plt.show()

# 4. Heatmap (Correlation)
plt.figure(figsize=(8, 6))
corr = df.iloc[:, 1:].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 5. Histogram (mAP50 Distribution)
plt.figure(figsize=(8, 5))
sns.histplot(df["mAP50"], bins=10, kde=True, color="purple")
plt.xlabel("mAP50")
plt.ylabel("Frequency")
plt.title("Distribution of mAP50")
plt.show()

# 6. Box Plot (Distribution of Precision per Class)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Precision (P)", color="cyan")
plt.xlabel("Precision (P)")
plt.title("Box Plot of Precision per Class")
plt.show()

# 7. Treemap (Instances per Class)
plt.figure(figsize=(8, 6))
sizes = df["Instances"]
labels = df["Class"]
squarify.plot(sizes=sizes, label=labels, alpha=0.7, color=sns.color_palette("husl", len(df)))
plt.axis("off")
plt.title("Instances per Class (Treemap)")
plt.show()

# 8. Waterfall Chart (Instances per Class)
df_sorted["Cumulative"] = df_sorted["Instances"].cumsum()

plt.figure(figsize=(10, 5))
bars = plt.bar(df_sorted["Class"], df_sorted["Instances"], color="blue", alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 2), ha="center", va="bottom")

plt.xticks(rotation=45)
plt.xlabel("Class")
plt.ylabel("Instances")
plt.title("Waterfall Chart of Instances per Class")
plt.show()

# 9. Gantt Chart (Class-wise Training Progress - Simplified)
# Create dummy training start & end times (adjust for real data)
df_sorted["Start"] = [datetime.datetime(2025, 3, 1) + datetime.timedelta(days=i * 2) for i in range(len(df_sorted))]
df_sorted["End"] = df_sorted["Start"] + pd.to_timedelta(df_sorted["Instances"], unit="D")

fig, ax = plt.subplots(figsize=(10, 6))
for i, (class_name, start, end) in enumerate(zip(df_sorted["Class"], df_sorted["Start"], df_sorted["End"])):
    ax.barh(class_name, (end - start).days, left=start, color=sns.color_palette("coolwarm", len(df_sorted))[i])

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.set_xlabel("Time")
ax.set_ylabel("Class")
plt.title("Gantt Chart of Training Progress")
plt.xticks(rotation=45)
plt.show()