import pandas as pd

df1 = pd.read_csv("final_dataset.csv", header=None)
df2 = pd.read_csv("live_data.csv", header=None)

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv("combined_dataset.csv", index=False, header=False)

print("New shape:", df.shape)