import pandas as pd
import matplotlib.pyplot as plt
import pycountry
file_path = "Task_1_dataset/API_SP.POP.TOTL_DS2_en_csv_v2_33112.csv"
df = pd.read_csv(file_path, skiprows=4)
country_codes = {
    country.alpha_3
    for country in pycountry.countries
}
df = df[df["Country Code"].isin(country_codes)]
population_2025 = df[["Country Name", "Country Code", "2025"]].copy()
population_2025 = population_2025.dropna(subset=["2025"])
top_10 = population_2025.sort_values(
    by="2025",
    ascending=False
).head(10)
top_10["Population_Millions"] = top_10["2025"] / 1_000_000
print("\nTop 10 Most Populated Countries in 2025:")
print(
    top_10[["Country Name", "2025"]]
    .to_string(index=False)
)
plt.figure(figsize=(12, 7))
bars = plt.bar(
    top_10["Country Name"],
    top_10["Population_Millions"]
)
for bar, value in zip(bars, top_10["Population_Millions"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        f"{value:.1f}M",
        ha="center",
        va="bottom",
        fontsize=9
    )
plt.title(
    "Top 10 Most Populated Countries in 2025",
    fontsize=16
)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Population (Millions)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    "top_10_population_2025.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()