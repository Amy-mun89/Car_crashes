import pandas as pd
import numpy as np

# read as object to avoid that pandas does any smart type conversion
# because want to store sample data in exactly the same format as original data
crashes = pd.read_csv("crashes.csv", dtype=object)
vehicles = pd.read_csv("vehicles.csv", dtype=object)
people = pd.read_csv("people.csv", dtype=object)

# keep only crashes from October 2017
dates_parsed = pd.to_datetime(crashes["CRASH_DATE"])
crashes = crashes[dates_parsed >= pd.to_datetime("2017-10-01")]

# get 10% sample of crashes
in_sample = np.random.rand(len(crashes)) < 0.1
crashes_sample = crashes[in_sample]

# get associated vehicles and people
included_rd_nos = set(crashes_sample["RD_NO"])
vehicles_sample = vehicles[vehicles["RD_NO"].apply(lambda r: r in included_rd_nos)]
people_sample = people[people["RD_NO"].apply(lambda r: r in included_rd_nos)]

# store
crashes_sample.to_csv("crashes.sample.csv", index=False)
vehicles_sample.to_csv("vehicles.sample.csv", index=False)
people_sample.to_csv("people.sample.csv", index=False)


file="All_Crashes_data.csv"
crash_data_raw = pd.read_csv(file)
crash_data_raw.info()
