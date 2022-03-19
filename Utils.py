import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Arrays to store details
method_name = []
model_score = []
dataset_path="datasets/houseprices/"


def load_data():
    df_train = pd.read_csv(dataset_path + "train.csv", index_col='Id')
    df_test = pd.read_csv(dataset_path + "test.csv", index_col='Id')
    main_data = pd.concat([df_train, df_test])
    numeric_columns = main_data.select_dtypes(include=np.number).columns.tolist()
    num_cols = [nc for nc in numeric_columns if main_data[nc].nunique()>20]

    df = main_data[num_cols]
    del df['SalePrice']
    num_cols = df.columns
    return main_data, df, num_cols


# Building model
def run_model(method, x_train,y_train):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    score = model.score(x_train, y_train)
    method_name.append(method)
    model_score.append(score)
    print(f"Model Accuracy : {score}")


# lets try to check the percentage of missing values,unique values,percentage of one catagory values and type against each column.
def statistics(df):
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum(), df[col].isnull().sum() * 100 / df.shape[0], df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Missing values', 'Percentage of Missing Values', 'Data Type'])
    stats_df.set_index('Feature', drop=True, inplace=True)
    stats_df.drop(stats_df[stats_df['Missing values'] == 0].index, axis=0, inplace=True)
    stats_df.sort_values('Percentage of Missing Values', ascending=False, inplace=True)
    return stats_df


# Function to display detected outliers
def outliers_statistics(df):
    stats = []
    for col in df.columns:
        stats.append((col, df[col].isnull().sum(), df[col].isnull().sum() * 100 / df.shape[0]))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Outliers', 'Percentage of Outliers'])
    stats_df.set_index('Feature', drop=True, inplace=True)
    stats_df.drop(stats_df[stats_df['Outliers'] == 0].index, axis=0, inplace=True)
    stats_df.sort_values('Percentage of Outliers', ascending=False, inplace=True)
    return stats_df


# Function to detect outliers with mean standard deviation and put nan instead of them
def std_mean(feature, df,std_thresh=2):
    upper_limit = df[feature].mean() + std_thresh*df[feature].std()
    lower_limit = df[feature].mean() - std_thresh*df[feature].std()
    index = df[(df[feature] > upper_limit) | (df[feature] < lower_limit)][feature].index
    df[feature].loc[index] = np.nan
    return df


# Function to detect outliers with median standard deviation and put nan instead of them
def std_median(feature, df,std_thresh=2):
    upper_limit = df[feature].median() + std_thresh*df[feature].std()
    lower_limit = df[feature].median() - std_thresh*df[feature].std()
    index = df[(df[feature] > upper_limit) | (df[feature] < lower_limit)][feature].index
    df[feature].loc[index] = np.nan
    return df


def prepare_data(df,main_data,num_cols, null_in_sale=50):
    # Taking out categorical features for remaning data
    new_df = main_data.copy()
    # new_df = new_df.loc[list(df.index), :]
    new_df[num_cols] = df.values

    # Dropping features in which more than 50 percent null values are present
    stat = statistics(new_df)
    new_df.drop(stat[stat["Percentage of Missing Values"] > null_in_sale].index, axis=1, inplace=True)

    # Filling null values with mode in features
    for i in statistics(new_df).index[1:]:
        new_df[i].fillna(new_df[i].mode()[0], inplace=True)

    # Encoding categorical feature
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    for i in new_df.select_dtypes(include=['O']).columns:
        new_df[i] = encoder.fit_transform(new_df[i])

    # Splitting data into train and test part
    train = new_df[new_df.SalePrice.notnull()]
    test = new_df[new_df.SalePrice.isnull()]
    #drop left overs
    train.dropna(inplace=True)
    # Separating data into input and output data
    X_train, y_train = train.drop("SalePrice", axis=1), train.SalePrice
    X_test, y_test = test.drop("SalePrice", axis=1), test.SalePrice
    
    return X_train, X_test, y_train, y_test


def set_data(df, num_cols):
    # Filling NAN with mean values
    for i in num_cols:
        df[i].fillna(df[i].mean(), inplace=True)
    
    # Filling null values with mode in categorical features
    for i in statistics(df).index[1:]:
        df[i].fillna(df[i].mode()[0], inplace=True)
    
    # Encoding categorical feature
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    for i in df.select_dtypes(include=['O']).columns:
        df[i] = encoder.fit_transform(df[i])
    return df


def split_data(df):
    # Splitting data into train and test part
    train = df[df.SalePrice.notnull()]
    test = df[df.SalePrice.isnull()]
    
    train.dropna(inplace=True)
    # Separating data into input and output data
    X_train, y_train = train.drop(["SalePrice", "outlier"], axis=1), train.SalePrice
    X_test, y_test = test.drop(["SalePrice", "outlier"], axis=1), test.SalePrice
    
    return X_train, X_test, y_train, y_test