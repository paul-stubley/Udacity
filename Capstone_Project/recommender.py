import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments
from numba import jit
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium import Choropleth, Circle, Marker, Popup
from folium.plugins import HeatMap, MarkerCluster
import math

### See below _fit function for Class definition ###

# Define the fit function outside of the class to allow numba, then use inside the class
@jit(nopython=True)
def _fit(ratings_mat, latent_features, learning_rate, iters, print_every):
        '''Function declared outside the Recommender class to allow Numba.  
        See internal method Recommender.fit() for full docs
        '''
        
        # Set up useful values to be used through the rest of the function
        n_users = ratings_mat.shape[0]  # number of rows in the matrix
        n_items = ratings_mat.shape[1] # number of item in the matrix
        num_ratings = ratings_mat.size  # total number of ratings in the matrix

        # initialize the user (n_user * k) and item (k * n_items) matrices  with random values
        user_mat = np.random.rand(n_users,latent_features)
        item_mat = np.random.rand(latent_features,n_items)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # header for running results
        print("Optimization Statistics")
        print("Iterations\t| Mean Squared Error\t")

        # for each iteration
        for i in range(iters):
            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-item pair
            for u in range(n_users):
                for m in range(n_items):
                    # if the rating exists
                    if np.isfinite(ratings_mat[u,m]):
                        # compute the error as the actual minus the predicted
                        err = ratings_mat[u,m]-np.dot(user_mat[u,:],item_mat[:,m])
                        # Keep track of the total sum of squared errors for the matrix
                        sse_accum += err**2
                        # update the values in each matrix in the direction of the gradient
                        user_mat[u,:] = user_mat[u,:] + learning_rate * 2 * err * item_mat[:,m].T
                        item_mat[:,m] = item_mat[:,m] + learning_rate * 2 * err * user_mat[u,:].T

            # print results for iteration
            if (i+1)/print_every == round((i+1)/print_every):
                print(i+1,'\t\t|', round(sse_accum / num_ratings,5))

        return user_mat, item_mat 



    
    
    
class Recommender():
    '''Class to hold a FunkSVD fitted recommender and associated information.  
    Predicts ratings based on interactions between generic 'users' and 'items'
    'items' could be, e.g., movies, restaurants, articles etc.
    See: https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)#Funk_MF
    N.B. Currently only integer ratings allowed

    Args:
    
        None
        
    Key methods (in order of  execution):
    
        self.set_user_item_matrix
        self.fit
        self.compare_train_to_predictions

    '''
    def __init__(self, ):
        '''No need to initialise'''
        
    def fit(self, latent_features=10, learning_rate=0.0001, iters=100, print_every=1):
        '''Numba-ised function to perform SVD of the user/item matrix into two matrices using FunkSVD
        Ouputs a user/latent-feature matrix and a latent-feature/item matrix
        Args:
             latent_features - (int) number of latent features in the FunkSVD fit
             learning_rate   - (float) learning rate
             iters           - (int) number of iterations
             print_every     - (int) print an output line every <print_every> iterations
             
        Returns:
            None 
            
        Sets attributes:
             self.user_mat - (numpy array) a user by latent feature matrix
             self.item_mat - (numpy array) a latent feature by item matrix
        '''
        self.user_mat, self.item_mat = _fit(np.array(self.ratings_mat)
                                , latent_features
                                , learning_rate
                                , iters
                                , print_every)
        
        self.user_df = pd.DataFrame(self.user_mat, index = self.ratings_mat.index
                                    , columns = ['k'+str(i+1) for i in range(latent_features)])
        self.item_df = pd.DataFrame(self.item_mat, columns = self.ratings_mat.columns
                                    , index =   ['k'+str(i+1) for i in range(latent_features)])

        # Create prediction matrix and clip to min-max rating
        self.ratings_mat_pred = pd.DataFrame(np.dot(self.user_mat, self.item_mat)
                                               , index = self.ratings_mat.index
                                               , columns = self.ratings_mat.columns) \
                                .clip(self.min_rating,self.max_rating)
        
        return None
    
    def get_user_item_matrix(self, df, user_id_col_name, item_id_col_name, rating_col_name):
        '''Sets the self.ratings_mat attribute as the user/item matrix
        Args:
            df               - (pd.Dataframe) containing at least columns for user_id, item_id, rating
            user_id_col_name - (string) name of column to use for user_id 
            item_id_col_name - (string) name of column to use for item_id 
            rating_col_name  - (string) name of column to use for rating 

        Returns:
            ratings_mat - (pd.Dataframe) the user-item matrix (users as rows
                                      , items as cols, ratings as values)
                        
        '''
        ratings_mat = df[[user_id_col_name, item_id_col_name, rating_col_name]] \
                            .groupby([user_id_col_name, item_id_col_name])[rating_col_name] \
                            .max() \
                            .unstack()
        
        return ratings_mat
    
    def set_user_item_matrix(self, df, user_id_col_name, item_id_col_name, rating_col_name):
        '''Sets the attribute self.ratings_mat to the user/item matrix
        Args:
            df               - (pd.Dataframe) containing at least columns for user_id, item_id, rating
            user_id_col_name - (string) name of column to use for user_id 
            item_id_col_name - (string) name of column to use for item_id 
            rating_col_name  - (string) name of column to use for rating 

        Returns:
            ratings_mat - (pd.Dataframe) the user-item matrix (users as rows
                                      , items as cols, ratings as values)
        '''
        self.ratings_mat = self.get_user_item_matrix(df
                                                     , user_id_col_name
                                                     , item_id_col_name
                                                     , rating_col_name) 
        self.reviews_df = df
        self.user_id_col_name = user_id_col_name
        self.item_id_col_name = item_id_col_name
        self.rating_col_name = rating_col_name
        self.max_rating = df[rating_col_name].max()
        self.min_rating = df[rating_col_name].min()
        print('Min rating: {}, max rating: {}'.format(self.min_rating,self.max_rating))

        return None
        
    def compare_train_to_predictions(self, test_df, plot=True):
        '''Compares the overlapping sections of the test ratings with the
            predicted ratings from FunkSVD
        
        Args:
            test_df - (pd.Dataframe) The test ratings df, one row per rating
                            , same column names as the train dataframe
            plot    - (bool) whether to plot the actual vs predicted ratings
            
        Returns:
            confusion_array - (np.ndarray) actual vs predicted ratings confusion matrix
            
        '''
        
        # Create the test user-item matrix
        ratings_mat_test = self.get_user_item_matrix(test_df
                                                , self.user_id_col_name
                                                , self.item_id_col_name
                                                , self.rating_col_name)
        
        # Get users in both train and test
        shared_users = set(self.ratings_mat.index).intersection(ratings_mat_test.index)
        # Get items in both train and test
        shared_items = set(self.ratings_mat.columns).intersection(ratings_mat_test.columns)

        
        # Get train_matrix cropped to size of test matrix
        ratings_mat_pred_cropped = self.ratings_mat_pred.loc[shared_users,shared_items]
        ratings_mat_test_cropped = ratings_mat_test.loc[shared_users,shared_items]
        
        # Confirm they have the same order of rows and columns
        err_string = "The {} in the predicted matrix and test matrix " \
                        +"have not been overlapped correctly inside this function"
        assert (ratings_mat_pred_cropped.index != ratings_mat_test_cropped.index) \
                .sum() == 0  , err_string.format("user indeces")
        assert (ratings_mat_pred_cropped.columns != ratings_mat_test_cropped.columns) \
                .sum() == 0 , err_string.format("item columns")
        
        # Compare overlaps between prediction and test by element-wise concatenation of two DFs
        # Concatenate element-wise
        pred =     ratings_mat_pred_cropped.applymap(lambda y: str(round(y)))
        test =  ratings_mat_test_cropped.fillna(-1).applymap(lambda y: str(round(y)))
        actual_vs_prediction = test.apply(lambda t : t+','+pred[t.name])
        
        # Get value counts of each true-prediction combination
        vc = actual_vs_prediction.melt().value.value_counts()
        
        # Fill in confusion array, of size dimension x dimension
        dimension = self.max_rating-self.min_rating+1
        confusion_array = np.zeros((dimension, dimension))
        indeces = vc.index.str.split(',',expand=True).to_list()
        
        for i,(t,p) in enumerate(indeces):
            if int(t) > -1: # Ignore NaNs (which have been filled as -1s)
                confusion_array[int(t)-1,int(p)-1] = vc[i]
        
        # Plots heatmap if required
        if plot:
            fig, ax = plt.subplots(1,2,figsize=(12,4))
            sns.heatmap(confusion_array, annot=True, fmt='.0f', ax=ax[0])
            plt.xticks(np.arange(self.min_rating,self.max_rating+1,1)-0.5
                      ,np.arange(self.min_rating,self.max_rating+1,1))
            plt.yticks(np.arange(self.min_rating,self.max_rating+1,1)-0.5
                      ,np.arange(self.min_rating,self.max_rating+1,1))
            ax[0].xaxis.tick_top()
            
            ### Need to get list of predictions and actuals ####
            
#             ax[1].hist(acts, density=True, alpha=.5, label='actual', bins=range(8));
#             ax[1].hist(preds, density=True, alpha=.5, label='predicted', bins=range(8));
#             ax[1].legend(loc=2, prop={'size': 15});
#             ax[1].xlabel('Rating');
#             ax[1].title('Predicted vs. Actual Rating');

        return confusion_array
    
    def predict_ratings_for_user(self, user_id, num_recs=-1):
        '''For a given user_id, returns a given number of recommendations
        Args:
            user_id  - (str or int) user_id to find recommendations for
            num_recs - (int) number of recommendations to supply, if blank, supply all.
        Return:
            series of recommendations - (pd.Series) item_id and predicted_rating
        '''
        
        total_num = self.ratings_mat_pred.shape[1]
        if num_recs == -1:
            num_recs = total_num
        predictions = self.ratings_mat_pred.loc[user_id] \
        
        not_rated_yet = self.ratings_mat.loc[user_id].isna().values
        
        # Remove the items already rated, sort and rename
        predictions = predictions.loc[not_rated_yet] \
                       .sort_values(ascending=False)[:num_recs] \
                       .rename('predicted_rating')
            
        return predictions

    def get_item_names(self, df_item, item_series, item_name_col, other_cols=[]):
        ''' For a given set of item ids, get their names from df_item, a
            dataframe containing the item_id column (defined in the 
            set_user_item_matrix method) and an item_name column, defined
            here.  Can optionally return other columns from df_item
        
        Args:
            df_item       - (pd.Dataframe) dataframe of items containing the item_id and item_name
            item_series   - (pd.Series) ordered Series with item ids as index
            item_name_col - (str) name of the item_name column
            other_cols    - (optional list) other columns from df_item to return with the output
            
        Return:
            recomms_df     - (pd.Dataframe) dataframe of ids, names, and any other columns from 
                           df_item that were specified
            
        '''
        
        # Join the id series with the df_item dataframe and return desired columns
        recomms_df = pd.DataFrame(item_series) \
                     .merge(df_item, on = self.item_id_col_name) \
                      [[self.item_id_col_name
                      , item_name_col
                      , *other_cols]] # Unpack other_cols into the slice
        
        return recomms_df
    
    
    def plot_locations(self, recomms_df, item_name_col, latitude_name, longitude_name, info = None ,search_string='', icon='cutlery'):
        '''Cluster plot the locations on a Folium map
        Args:
            items_df       - (pd.DataFrame) ideally output of self.get_item_names
            item_name_col  - (string) Name of the item column
            latitude_name  - (string) Name of the latitude column
            longitude_name - (string) Name of the longitude column
            info           - (string) Popup additional info, allowed values:
                                            'predicted_rating'
                                            'similarity'
            search_string  - (string) String to append to the front of the item name 
                                in the popup, e.g. a city name to add to the search, like "toronto"
                                item name will be appended two it, with spaces replaced by "+"
                                should be '+' separated if supplying multiple search terms
            icon           - (optional string) icon style, see `help(folium.Icon)` for more info
        Returns:
            folium_map     - (folium.Map) Output Folium map
        '''
        
        
        center_location= recomms_df[latitude_name].mean(),recomms_df[longitude_name].mean()

        m = folium.Map(location=center_location, tiles='cartodbpositron', zoom_start=11)

        search_string = '<a target="_blank" href="http://www.google.com/search?q='+search_string+'+'
        
        # create colors for ratings markers
        colors = ['darkred','red','orange','darkgreen','lightgreen']
        ratings_scale = self.max_rating - self.min_rating + 1 
        # expand if ratings have more than 5 options
        colors = ['darkred']*(ratings_scale-5) + colors
        
        ### Add points to the map

        # If there are more than 50 recommendations, plot as clusters, else plot them all
        if len(recomms_df) > 50:
            mc = MarkerCluster()
        else:
            mc = m
            
          

        for idx, row in recomms_df.iterrows():
            if not math.isnan(row[longitude_name]) and not math.isnan(row[latitude_name]):
                
                # Create popup text inc. link, round the rating/similarity appropriately
                if info:
                    color = colors[round(row[info]-1)]
                    if info == 'similarity':
                        pop_num = str(round(row[info],2))
                    else:
                        pop_num = str(round(row[info]))
                        
                    popup_string = '\n'+info.replace('_',' ').title() +': '+ pop_num
                                     
                else:
                    popup_string = ''
                    color = 'darkred'
                                         
                popup = folium.Popup(search_string + row[item_name_col].replace(' ','+')+'">' \
                                     +row[item_name_col]+'</a><br>'+popup_string , max_width=300)
                                         
                # Add point to chart or cluster
                mc.add_child(Marker([row[latitude_name], row[longitude_name]]
                                   , popup = popup
                                   , icon=folium.Icon(  color = color
                                                      , icon = icon
                                                      , prefix = 'fa'
                                                      , icon_color = 'white')
                                    )
                            )

        # Finish off the cluster plot if necessary        
        if len(recomms_df) > 50:
            m.add_child(mc)

        return m
        
    def get_top_reviewed_items(self, min_number_of_reviews = 0):
        '''Returns the ordered list of items, sorted by 
        Args: 
            min_number_of_reviews - (int) dismiss items with fewer than 
                                    this number of reviews
        Returns: 
            ordered_df - (pd.DataFrame) ordered list of item_ids
        '''
        mean_ratings = self.reviews_df[[self.user_id_col_name
                                        , self.item_id_col_name
                                        , self.rating_col_name]]      \
                        .groupby(self.item_id_col_name)[self.rating_col_name] \
                        .aggregate({np.mean, np.size})
        
        ordered_df = mean_ratings[mean_ratings['size']>=min_number_of_reviews] \
                      .sort_values(by='mean',ascending=False)
    
        return ordered_df
    
    def get_similar_items(self, item_id, n=-1):
        '''Returns the n most similar items to item_id
        Args: 
            item_id - (int or string) item id
            n       - (int) number of similar items to return, if -1 return all
        Returns
            similarities - (pd.Series) item_ids as index
        '''
        # Find number of items to return if n is not given
        if n==-1:
            n = self.item_df.shape[1]-1
        
        # Normalise each items column to allow cosine similarity to be found
        normed = self.item_df.apply(lambda x: x/np.linalg.norm(x))
        # Take dot product to find similarites
        similarities = np.dot(np.transpose(normed[item_id]),normed)
        # Add item_ids back to similarities, sort descending, and clip to n
        similarities = pd.Series(similarities, index=self.item_df.columns) \
                         .sort_values(ascending=False)[1:n+1] \
                         .rename('similarity')
        
        return  similarities

        
        
        
        
    
