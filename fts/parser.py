import pandas as pd


data = pd.read_csv("./News.csv")

lenDocs  = len(data['content'])
