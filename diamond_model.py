import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split 
import json
import pickle



df = pd.read_csv(r'C:\Users\HP\OneDrive\İş masası\Diamonds\Diamonds Prices2022.csv')
df_normalize = pd.read_csv(r'C:\Users\HP\OneDrive\İş masası\Diamonds\df_norm.csv')
df_normalize = df_normalize.drop(columns = ['Unnamed: 0'])

pd.set_option('display.max_columns', None)

X = df_normalize.drop(columns = ['price'])
y = df_normalize['price']


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

# Create an XGBoost regression model
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=5,
    learning_rate=0.2,
    random_state=42,
    min_child_weight =  4,
    subsample = 0.7,
    colsample_bytree = 1.0
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Evaluate the model's performance using regression metrics
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')
r2 = r2_score(y_test, y_pred)
print(f'R-squared (R2) Score: {r2}')
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')


#Let's predict two diamond price and check model as manually
new = np.array([[-1.240439,-0.664790,0.728035,0.017228,-0.176149 , 0.764210,-0.232610,
        1.996793,1.165418,-0.183425,0.764467,1.176583,-1.364482,1.582252,
       -1.367087,1.577530, -0.387590, 1.652025]])

new1 = np.array([[-1.029370,1.504235,0.728035,0.017228,-0.176149,0.764210,
                  -0.879024,1.576406,-0.147787,-0.183425,0.764467,-0.154689,
                  1.080352,0.253795,1.077479,0.264330,-0.38759,0.206536]])

y_pred = xgb_model.predict(new)
y_pred1 = xgb_model.predict(new)

#Let's compare model prediction and actual values
print(f'·Model Predict: {round(float(y_pred),5)} | · Actual Value: {round(y.iloc[1],5)}')
print(f'·Model Predict: {round(float(y_pred1),5)} | · Actual Value: {round(y.iloc[4],5)}')

with open("xgbmodel.pkl", 'wb') as file:
    pickle.dump(xgb_model, file)

mean = df['price'].mean()
std = df['price'].std()

standardized = {"mean":mean , "std" :std}
# Open the JSON file in write mode
with open("standardized.json", 'w') as file:
    json.dump(standardized, file)

with open('standardized.json','rb') as json_file:
    json_file = json.load(json_file)





