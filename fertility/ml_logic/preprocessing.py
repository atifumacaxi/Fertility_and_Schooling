#Importing libraries
import pandas as pd
import numpy as np

#Loading fertility dataset
df_fertility = pd.read_csv('fertility/data/fertility_rate.csv')

#Loading schooling dataset
df_schooling = pd.read_csv('fertility/data/mean-years-of-schooling-long-run.csv', sep=';')

def preprocessing_features(df_schooling:pd.DataFrame, df_fertility:pd.DataFrame) -> pd.DataFrame:
    #Renaming "Entity" column to "Country" in the schooling dataset
    df_schooling = df_schooling.rename(columns={'Entity': 'Country'})
    #Transforming the years columns to a single "Year" column containing all years in the fertility dataset
    df_fertility = pd.melt(df_fertility,
                        id_vars=["Country",],
                        var_name="Year", value_name="fertility")

    #Sorting values by year in the fertility dataset
    df_fertility = df_fertility.sort_values(["Country"])
    df_fertility.sort_values("Year")

    #Transforming the "Year" column values into integers and removing everything below the year 1960
    df_fertility = df_fertility.astype({'Year':'int'})
    df_fertility.drop(df_fertility[df_fertility['Year']<=1959].index, inplace = True)

    #Removing every "_World" value in the "Country" column since we won't be using them
    df_fertility.drop(df_fertility[df_fertility['Country'] == '_World'].index, inplace = True)

    #Merging both datasets into one, each row and column with their respective matching values
    df = df_fertility.merge(df_schooling, how='inner', on=('Country', 'Year'))

    #Order by Year and Coutry, cause we are dealing with time series
    df = df.sort_values(['Year', 'Country'])

    # Select
    df = df[['Year', 'Country', 'Code', 'fertility', 'avg_years_of_schooling']].copy()

    return df
