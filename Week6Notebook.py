# Databricks notebook source
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# COMMAND ----------

dbutils.widgets.text("output", "","")
dbutils.widgets.get("output")
FilePath = getArgument("output")

 

# COMMAND ----------

dbutils.widgets.text("filename", "","")
dbutils.widgets.get("filename")
filename = getArgument("filename")
storage_account_name = "storageaccount7766"
storage_account_access_key = "8YxCdC34sDcBWPDR4cOzJWbJ1HlXcLpSB5Jt33WTWt8+I7H2rvfGPI4uHBYiKwF0zGqAuLRDJeOF358aTOEVGw=="

# COMMAND ----------

spark.conf.set(
"fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
storage_account_access_key)

# COMMAND ----------

file_location = "wasbs://reviewblob@storageaccount7766.blob.core.windows.net"+FilePath+"/"+filename
print(file_location)
file_type = "csv"

# COMMAND ----------

df = spark.read.format(file_type).option("inferSchema", "true").load(file_location)
df.show()

# COMMAND ----------

train=df.toPandas()

# COMMAND ----------

x=train.iloc[2:len(train),2:4].values

y=train.iloc[2:len(train),:1].values

# COMMAND ----------

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
reg = LinearRegression()
reg.fit(x_train,y_train)


# COMMAND ----------

file_name = 'House_Price.pkl'
pkl_file = open(file_name, 'wb')
model = pickle.dump(reg, pkl_file)


# COMMAND ----------

pkl_file = open(file_name, 'rb')
model_pkl = pickle.load(pkl_file)
y_pred = model_pkl.predict(x_test)
print("prediction",y_pred)



