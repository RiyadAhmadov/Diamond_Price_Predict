import json
import pickle
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Create a FastAPI app instance
app = FastAPI()

# Load the pickled model using a relative file path
with open('xgbmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the endpoint to make predictions
@app.get("/predict/")
async def predict(
    carat: float = Query(..., description="Carat"),
    cut: str = Query(..., description="Cut"),
    color: str = Query(..., description="Color"),
    clarity: str = Query(..., description="Clarity"),
    depth: float = Query(..., description="Depth"),
    table: float = Query(..., description="Table"),
    x: float = Query(..., description="X dimension"),
    y: float = Query(..., description="Y dimension"),
    z: float = Query(..., description="Z dimension")
):

    # Create a DataFrame with the input data
    data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    } , index = [1])

    
    # Calculate new columns
    data['Volume'] = data['x'] * data['y'] * data['z']
    data['density'] = data['carat'] / data['Volume']
    data['depth_percentage'] = data['depth'] / (data['x'] + data['y'])
    data['two_sep_color'] = data['color'].apply(lambda x: 1 if x in ['H', 'I', 'J'] else 0)
    data['two_sep_clarity'] = data['clarity'].apply(lambda x: 1 if x in ['I1', 'SI1', 'SI2', 'VS2'] else 0)
    data['table_flat'] = data['table'].apply(lambda x: 0 if x < 82.0 and x > 69.0 else 1)

    bins = [42.0, 50.0, 57.0, 64.0, 80.0]
    labels = ['42.0-50.0', '50.0-57.0', '57.0-64.0', '64.0+']
    data['depth_bin'] = pd.cut(data['depth'], bins=bins, labels=labels)
    map_dict1 = {'42.0-50.0': 3, '50.0-57.0': 1, '57.0-64.0': 2, '64.0+': 4}
    data['depth_label'] = data['depth_bin'].map(map_dict1)

    bins = [42.0, 56.0, 69.0, 82.0, 96.0]
    labels = ['42.0-56.0', '56.0-69.0', '69.0-82.0', '82.0+']
    data['table_bin'] = pd.cut(data['table'], bins=bins, labels=labels)
    map_dict1 = {'42.0-56.0': 2, '56.0-69.0': 3, '69.0-82.0': 1, '82.0+': 4}
    data['table_label'] = data['table_bin'].map(map_dict1)

    data['Length_Width_Ratio'] = data['x'] / data['y']
    
    bins = [0.1, 1.3, 2.7, 4, 5.02]
    labels = ['0.1-1.3', '1.3-2.7', '2.7-4.5', '4.5+']
    data['carat_bin'] = pd.cut(data['carat'], bins=bins, labels=labels)
    map_dict1 = {'0.1-1.3':1,'1.3-2.7':2,'2.7-4.5':3,'4.5+':4}
    data['carat_label'] = data['carat_bin'].map(map_dict1)
    data['log_carat_label'] = np.log(data['carat_label'])
    


    categorical_columns = ['Length_Width_Ratio', 'depth_label', 'table_label']
    for col in categorical_columns:
        data[col] = data[col].astype('category').cat.codes


    
    data['pow_Length_Width_Ratio'] = data['Length_Width_Ratio']**2

    data['pow_depth_label'] = data['depth_label']**2
    data['pow_table_label'] = data['table_label']**2
    data['sqrt_depth'] = data['depth']**2
    data['sqrt_table'] = data['table']**2
    data['log_depth'] = np.log(data['depth'])
    data['log_table'] = np.log(data['table'])

    low_corr_cols = ['depth',
                    'table',
                    'cut_num',
                    'cut_Fair',
                    'cut_Good',
                    'cut_Ideal',
                    'cut_Premium',
                    'cut_Very Good',
                    'color_D',
                    'color_E',
                    'color_F',
                    'color_G',
                    'color_H',
                    'color_I',
                    'color_J', 
                    'clarity_I1',
                    'clarity_IF',
                    'clarity_SI1',
                    'clarity_SI2',
                    'clarity_VS1',
                    'clarity_VS2',
                    'clarity_VVS1',
                    'clarity_VVS2']

    cut_values = ['Ideal','Good','Very Good','Fair','Premium']
    color_values = ['D','E','F','G','H','I','J'] 
    clarity_values = ['I1','IF','SI1','SI2','VS1','VS2','VVS1','VVS2']
       
    for i in cut_values:
        data[f'cut_{i}'] = data['cut'].apply(lambda x: 1 if x == i else 0)

    for i in color_values:
        data[f'color_{i}'] = data['color'].apply(lambda x: 1 if x == i else 0)

    for i in clarity_values:
        data[f'clarity_{i}'] = data['clarity'].apply(lambda x: 1 if x == i else 0)


    map_dict = {'Ideal':1,'Good':2,'Very Good':3,'Fair':4,'Premium':5}
    data['cut_num'] = data['cut'].map(map_dict)
    

    #Let's use PCA for creating new features and dimesion reduction
    X = data[low_corr_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = 1
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    data['pca_1'] = X_pca[:,0]

    scaler = StandardScaler()
    X = data
    scaler.fit(X)
    X_standardized = scaler.transform(X)
    data_normalize = pd.DataFrame(X_standardized ,columns = data.columns)


    # Perform model predictions
    predictions = model.predict(data_normalize[['carat', 'two_sep_color', 'two_sep_clarity', 
                                  'table_flat','depth_label', 'table_label', 'density', 'depth_percentage',
                                  'Length_Width_Ratio', 'pow_depth_label', 'pow_table_label',
                                  'pow_Length_Width_Ratio', 'sqrt_depth', 'sqrt_table', 'log_depth',
                                  'log_table', 'log_carat_label', 'pca_1']])

    # Interpret predictions as needed, for example, you can round the prediction
    rounded_predictions = np.round(predictions, 2)

    # Return the prediction result
    prediction_result = f"This diamond price is {rounded_predictions[0]} !!!"
    
    return {"prediction": prediction_result}

# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5011,
        log_level="debug",
    )