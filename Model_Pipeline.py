import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = 'Model.pkl'
PIPELINE_FILE = 'Pipeline.pkl'

def build_pipeline(cat_attribs, num_attribs):
    # Numerical pipeline

    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
    ])

    # Categorial pipeline

    cat_pipeline = Pipeline([
    ('onehoteEncoding', OneHotEncoder(handle_unknown='ignore'))
    ])

    # FULL PIPELINE

    full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):
    df = pd.read_csv('housing.csv')


    df['income_split'] = pd.cut(df['median_income'], bins= [0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels= [1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in split.split(df, df['income_split']):
        df.loc[test_idx].drop('income_split', axis=1).to_csv('test_data_with_label.csv', index=False)
        df.loc[test_idx].drop(['income_split', 'median_house_value'], axis=1).to_csv('input.csv', index=False)
        df_train = df.loc[train_idx].drop('income_split', axis= 1)

    housing = df_train.copy()

    # Now we will separate features and labels

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis = 1).copy()

    # Now we will specify numerical attributes and categorial attirbutes

    cat_attribs = ['ocean_proximity']
    num_attribs = housing_features.drop('ocean_proximity', axis=1).columns.tolist()

    # Now we will fit transform this data in our pipeline
    pipeline = build_pipeline(cat_attribs, num_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    # Now we will train our model 

    # Let's first use GridSearchCV to get the best parameters for our model
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
    "n_estimators": [100, 200, 300],     # number of trees in the forest
    "max_depth": [None, 10, 20],         # how deep each tree can go
    "min_samples_split": [2, 5, 10],     # min samples needed to split a node
    "min_samples_leaf": [1, 2, 4],       # min samples at the end leaf
    "max_features": ["auto", "sqrt"]     # how many features to consider at each split
    }

    grid_search = GridSearchCV(estimator = rf, param_grid= param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2 )

    grid_search.fit(housing_prepared, housing_labels)
    print(grid_search.best_params_)

    # Now we will train the model on the best parameters found by grid search CV
    
    model = RandomForestRegressor(random_state=42, **grid_search.best_params_)
    model.fit(housing_prepared, housing_labels)

    # Now we will save the pipelne and the model using joblib so that we don't need to do this step again and again

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print('Model trained and saved <3')

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv('output.csv', index=False)

    print('Inference completed. Results saved in output.csv <3')