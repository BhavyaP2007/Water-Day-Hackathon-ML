import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np
root_dir = Path(__file__).parent
root_dir = str(root_dir).replace("\\","/")
df_train = pd.read_csv(root_dir+"/dataset/train.csv")
df_train = df_train[df_train["Residents"]>=0]
df_train["type"] = "train"
df_test = pd.read_csv(root_dir+"/dataset/test.csv")
df_test["type"] = "test"
df = pd.concat([df_train,df_test],axis=0).reset_index()
# data cleaning and processing
df["Temperature"] = df["Temperature"].interpolate()
a = df.loc[df["Water_Consumption"]==192.5].loc[:,"Apartment_Type"]
df["Appliance_Usage"] = df["Appliance_Usage"].interpolate()
df = df.replace({a.tolist()[0]:np.nan})
df["Apartment_Type"] = df["Apartment_Type"].replace({np.nan:"Missing Home"})
encoder = LabelEncoder()
df["Apartment_Type"] = encoder.fit_transform(df["Apartment_Type"])
def categorize_income(income):
    if income=="Rich":
        return 3
    elif income=="Upper Middle":
        return 2
    elif income=="Middle":
        return 1
    elif income=="Low":
        return 0
    else:
        return -1
df["Income_Level"] = df["Income_Level"].apply(categorize_income)
encoder = LabelEncoder()
df["Amenities"] = encoder.fit_transform(df["Amenities"])
for i in df[df["type"]=="train"].index:
    try:
        df.loc[i,"Humidity"] = float( df.loc[i,"Humidity"])
    except:
        df.drop(i,axis=0,inplace=True) 
for i in df[df["type"]=="test"].index:
    try:
        df.loc[i,"Humidity"] = np.nan
    except:
        pass 
df["Humidity"] = df["Humidity"].infer_objects()        
df["Humidity"] = df["Humidity"].interpolate()                
X_train = df[df["type"]=="train"].iloc[:,2:12]
Y_train = df[df["type"]=="train"].iloc[:,12]
X_test = df[df["type"]=="test"].iloc[:,2:12]
X_timestamp = df[df["type"]=="test"].iloc[:,1:2]
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,accuracy_score

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(Y_train,model.predict(X_train))
new_df = pd.DataFrame(data=X_timestamp)
new_df["Water_Consumption"] = y_pred
print(new_df)
new_df.to_csv(root_dir+"/dataset/prediction.csv",index=False)
print(max(0,100- np.sqrt(mae)))
