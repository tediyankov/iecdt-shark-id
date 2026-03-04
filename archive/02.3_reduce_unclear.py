import pandas as pd
import random

# Set random seed for reproducibility
random.seed(42)

# Load the test set labels
df = pd.read_csv("test_set_labels.csv")

print(f"Original counts:")
print(df['species'].value_counts())
print()

# Separate unclear/other from the rest
unclear = df[df['species'] == 'unclear/other']
other_species = df[df['species'] != 'unclear/other']

print(f"Unclear/other images: {len(unclear)}")

# Randomly sample 500 from unclear/other
unclear_sampled = unclear.sample(n=500, random_state=42)

# Combine back together
df_new = pd.concat([other_species, unclear_sampled], ignore_index=True)

# Save to new file
df_new.to_csv("test_set_labels.csv", index=False)

print(f"\nNew counts:")
print(df_new['species'].value_counts())
print(f"\nTotal images: {len(df_new)}")
print("\nUpdated test_set_labels.csv saved!")
