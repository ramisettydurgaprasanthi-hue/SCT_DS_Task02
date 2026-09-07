import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("dataset/train.csv")
print("Original Dataset Shape:", df.shape)
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())
print("\nDuplicate Rows Before Cleaning:")
print(df.duplicated().sum())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])
print("\nDataset Shape After Cleaning:", df.shape)
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())
print("\nCleaned Dataset:")
print(df.head())
# Counting survived and non-survived passengers
survival_counts = df["Survived"].value_counts()
print("\nSurvival Counts:")
print(survival_counts)
survival_rate = df["Survived"].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.2f}%")
import os
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.countplot(
    data=df,
    x="Survived"
)
plt.title("Titanic Passenger Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(
    "outputs/survival_count.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# Calculate survival rate by gender
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
print("\nSurvival Rate by Gender:")
print(gender_survival)
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df,
    x="Sex",
    y="Survived"
)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
for i, value in enumerate(gender_survival):
    plt.text(
        i,
        value + 0.03,
        f"{value:.1f}%",
        ha="center"
    )
plt.tight_layout()
plt.savefig(
    "outputs/survival_by_gender.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# Calculate survival rate by passenger class
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
print("\nSurvival Rate by Passenger Class:")
print(class_survival)
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df,
    x="Pclass",
    y="Survived"
)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
for i, value in enumerate(class_survival):
    plt.text(
        i,
        value + 0.03,
        f"{value:.1f}%",
        ha="center"
    )
plt.tight_layout()
plt.savefig(
    "outputs/survival_by_class.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
print("\nAge Statistics:")
print(df["Age"].describe())
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="Age",
    bins=20,
    kde=True
)
plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(
    "outputs/age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
print("\nAverage Age by Survival Status:")
print(df.groupby("Survived")["Age"].mean())
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="Survived",
    y="Age"
)
plt.title("Age Distribution by Survival Status")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(
    "outputs/age_vs_survival.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
print("\nFare Statistics:")
print(df["Fare"].describe())
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="Fare",
    bins=30,
    kde=True
)
plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(
    "outputs/fare_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
numerical_data = df[
    ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
]
correlation_matrix = numerical_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap of Titanic Numerical Variables")
plt.tight_layout()
plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
print("\n" + "=" * 50)
print("FINAL EDA SUMMARY")
print("=" * 50)
print(f"\nTotal passengers analyzed: {len(df)}")
print(f"Overall survival rate: {df['Survived'].mean() * 100:.2f}%")
print("\nSurvival rate by gender:")
print(df.groupby("Sex")["Survived"].mean().mul(100).round(2))
print("\nSurvival rate by passenger class:")
print(df.groupby("Pclass")["Survived"].mean().mul(100).round(2))
print("\nAverage age by survival status:")
print(df.groupby("Survived")["Age"].mean().round(2))
print("\nAverage fare by passenger class:")
print(df.groupby("Pclass")["Fare"].mean().round(2))