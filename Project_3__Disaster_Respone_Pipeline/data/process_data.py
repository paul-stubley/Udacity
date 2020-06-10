import sys
import pandas as pd
from  sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''function to read in the two csvs
    Args:
        messages_filepath (str) - filepath to csv
        categories_filepath (str) - filepath to csv
    Returns:
        pandas.DataFrame - joined df of two inputs
    '''
    messages = pd.read_csv(messages_filepath).set_index('id')
    categories = pd.read_csv(categories_filepath).set_index('id')
    return messages.join(categories)


def clean_data(df):
    '''function to clean the data
    Args:
        df (pandas.DataFrame) - data to clean
    Returns:
        pandas.DataFrame - cleaned data
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # pull out category names
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.rename_axis('categories', axis=1, inplace = True)

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1].astype(int)
    
    # There are some rows with related = 2, change to related = 1
    categories.related[categories.related==2]=1
    # If there are no messages flagged with a certain flag, this will break some classifiers
    # So dropping any columns where there are no flags at all
    categories = categories.loc[:,categories.describe().loc['max']!=0]

    # Drop non-dummied category column
    df = df.drop('categories', axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    '''function to save the data to a local DB,
       at location <database_filename>, with table name 'disaster_response'
    Args:
        df (pandas.DataFrame) - data to save
        database_filename (str) - database name
    Returns:
        None
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')  

def save_csv(df, csv_filename):
    '''function to save the data to json blob,
       at location <csv_filename>
    Args:
        df (pandas.DataFrame) - data to save
        csv_filename (str) - csv filename
    Returns:
        None
    '''
    df.to_csv(csv_filename)   

def main():
    '''function to ETL the data, requires 3 arguments in the following order:
    Args:
        disaster_messages (str) - filepath to a csv
        disaster_categories (str) - filepath to a csv
        saved_data_filepath (str) - filepath to save the output DB/json to
    Returns:
        None
    '''
 
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, saved_data_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        # For local database
        print('Saving data...\n    DATABASE: {}'.format(saved_data_filepath))
        save_data(df, saved_data_filepath)
        # For json to use with Heroku, not functioning
        #print('Saving data...\n    FILE: {}'.format(saved_data_filepath))
        #save_csv(df, saved_data_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()