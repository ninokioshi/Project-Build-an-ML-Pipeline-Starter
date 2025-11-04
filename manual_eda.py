import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned sample data
df = pd.read_csv("clean_sample.csv")

# Print a quick summary
print("=== Dataset Summary ===")
print(df.describe(include='all'))

# Generate correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("docs/04_eda_visual.png")
plt.close()

print("\nEDA visualization saved to docs/04_eda_visual.png")
