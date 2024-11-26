import pandas as pd

df = pd.read_excel("/home/sunboy/CAMs/multi_label/data/valsets.xlsx",index_col=0)

f = open("val.txt","w")

for img in df.index:
    f.write(f"{img}\n")
f.close()





