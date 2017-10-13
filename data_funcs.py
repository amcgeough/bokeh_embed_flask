
# -*- coding: utf-8 -*-
"""
Conatins functions needed to clean COSA issues files

@author: Naser Monsefi
"""
import os.path
import pandas as pd
import numpy as np
from re import search
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import nltk
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def match(small_list, big_list):
   """Returns indexes in big list for wher small list values are
   """
   return [big_list.index(x) for x in small_list if x in big_list]


def make_R(Z, dend_threshold, labels):
   """
   Returns the dendrogram stast for the linkage matrix Z

   Parameters
   ----------
   Z : ndarray
       The linkage matrix encoding the hierarchical clustering to
       render as a dendrogram. See the ``linkage`` function for more
       information on the format of ``Z``.

   dend_threshold : int
       Color threshold for dendrogram function
   labels : list
       lis t of lables for each leaf

   Returns
   -------
   R : dict
       A dictionary of data structures computed to render the
       dendrogram. Its has the following keys:
       ``'color_list'``
         A list of color names. The k'th element represents the color of the
         k'th link.
       ``'icoord'`` and ``'dcoord'``
         Each of them is a list of lists. Let ``icoord = [I1, I2, ..., Ip]``
         where ``Ik = [xk1, xk2, xk3, xk4]`` and ``dcoord = [D1, D2, ..., Dp]``
         where ``Dk = [yk1, yk2, yk3, yk4]``, then the k'th link painted is
         ``(xk1, yk1)`` - ``(xk2, yk2)`` - ``(xk3, yk3)`` - ``(xk4, yk4)``.
       ``'ivl'``
         A list of labels corresponding to the leaf nodes.
       ``'leaves'``
         For each i, ``H[i] == j``, cluster node ``j`` appears in position
         ``i`` in the left-to-right traversal of the leaves, where
         :math:`j < 2n-1` and :math:`i < n`. If ``j`` is less than ``n``, the
         ``i``-th leaf node corresponds to an original observation.
         Otherwise, it corresponds to a non-singleton cluster.
       ``'Clusters'``
   """
   # Creates dendrogram and store data
   R = dendrogram(
           Z,
           no_plot=True,
           color_threshold=dend_threshold,
           above_threshold_color="#000000",
           labels=labels,
           )
   clusters = fcluster(Z, dend_threshold, criterion='distance')
   # Infamous match function
   cluster_index = match(R['ivl'], labels)
   #    [list(df_tfidf.index).index(x) for x
   #                 in R['ivl'] if x in list(df_tfidf.index)]
   R['Clusters'] = [clusters[x] for x in cluster_index]
   return R


def make_Z(df_tfidf):
   """
   Returns linkage matrix for specific tfidf data set

   Parameters
   ----------
   df_tfidf : DataFrame
       A tdidf matrix that will be used to create linkage matrix base on
       cosine similarity
   Returns
   -------
   Z : ndarray
       The hierarchical clustering encoded as a linkage matrix. The indices
       are converted from 1..N to 0..(N-1) form and stored in Z[:,1] and
       Z[:,2] while each row represents the two nodes that had to be joined.
       Each cluster gets a new index going from n + 1. A cluster with an index
       less than n corresponds to one of the original observations. The
       distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2].
       A fourth column Z[:,3] is added where Z[i,3] is represents the number
       of original observations (leaves) in the non-singleton cluster i.

   Notes
   -----
   Can be merged with make_tfidf function
   """
#    dist = 1 - cosine_similarity(df_tfidf.drop(['CASE_NUMBER'], axis=1))
#
#    idx = df_tfidf['CASE_NUMBER'].tolist()
#
#    df_dist = pd.DataFrame(data=dist, index=idx, columns=idx)
#    df_dist[df_dist < 0.00000000001] = 0  # convert negative distances to zero
#
#    # generate the linkage matrix
#    Z = linkage(df_dist, 'ward')
   Z = linkage(df_tfidf.drop(['CASE_NUMBER'], axis=1),
               method='ward', metric='euclidean')
   return Z


def create_universe_Z(CODA_export_file, issues_df, universe_issues_list, k):
   """
   Check Universe_Z and update it if it is old. It keeps a record of input
   file timestamp and if it is changed creates a new universe Z

   Parameters
   ----------
   CODA_export_file : str
   CODA export csv file location to check the timestamp with

   issues_df : DataFrame
   All of issues that the universe is built from.

   universe_issues_list : list
   The issues list for making the universe
   """
   time_stamp = os.path.getmtime(CODA_export_file)

   def make_universe_Z():
       with open("run_data/interim/CODA_Export_time_stamp.txt", "w") as f:
           f.write(str(time_stamp))
       df_tfidf = make_tfidf(issues_df[issues_df['CASE_NUMBER'].isin(universe_issues_list)])
       Z = make_Z(df_tfidf)
       # line for exporting universe tfidf matrix, Z and clusters
       # comment for disabling
       # df_tfidf.drop(["CASE_NUMBER"], axis=1).to_csv("../data/interim/Universe_tfidf.csv", sep=",")
       pd.DataFrame(Z).to_csv("run_data/interim/Universe_Z.csv", sep=",", index=False)
       dend_threshold = Z[-k, 2]
       universe_clusters = fcluster(Z, dend_threshold, criterion='distance')
       pd.DataFrame({"Issues": df_tfidf.CASE_NUMBER,
                     "Clusters": universe_clusters}).to_csv("run_data/interim/Universe_clusters.csv", sep=",", index=False)
       return Z
   if os.path.isfile("run_data/interim/CODA_Export_time_stamp.txt"):
       with open("run_data/interim/CODA_Export_time_stamp.txt", "r") as f:
           old_time_stamp = float(f.read())
       if time_stamp == old_time_stamp:
           Z = pd.read_csv("run_data/interim/universe_Z.csv").values
       else:
           Z = make_universe_Z()
   else:
       Z = make_universe_Z()
   return Z


def load_issues(input_file):
   """ Function for loading raw file in Pandas and formatting them.
   """
   df = pd.read_csv(input_file)
   # Fix problem with special characters
   df = df.rename(columns={'\xef\xbb\xbfCASE_NUMBER': 'CASE_NUMBER'})
   # Fix problem with special characters
   df = df.rename(columns={'\ufeffCASE_NUMBER': 'CASE_NUMBER'})
   # Remove dupliates
   df = df.drop_duplicates()
   # Format columns into correct type
   df['CASE_NUMBER'] = df['CASE_NUMBER'].astype(str)
   df['DETAILED_DESCRIPTION'] = df['DETAILED_DESCRIPTION'].astype(str)
   #df['ISSUE_SUMMARY'] = df['ISSUE_SUMMARY'].astype(str)
   #df['ATTRIBUTE_DOCUMENT_HEADER'] = df['ATTRIBUTE_DOCUMENT_HEADER'].astype(str)
   #df['ISSUE_DETECTION_POINT'] = df['ISSUE_DETECTION_POINT'].astype(str)
   #df['SOURCE_SYSTEM'] = df['SOURCE_SYSTEM'].astype(str)
   #df['CASE_STATUS'] = df['CASE_STATUS'].astype(str)
   # Creating pandas index
   df['idx'] = df['CASE_NUMBER']
   df = df.set_index('idx')
   # idx = df['CASE_NUMBER'].tolist() # no sure if this is needed
   for i in range(0, len(df)):
       df['DETAILED_DESCRIPTION'][i] = df['DETAILED_DESCRIPTION'][i].replace("\x96", "")

   #for i in range(0, len(df)):
    #   df['ISSUE_SUMMARY'][i] = df['ISSUE_SUMMARY'][i].replace("\x96", "")

   #for i in range(0, len(df)):
    #   df['SOURCE_SYSTEM'][i] = df['SOURCE_SYSTEM'][i].replace("\x96", "")

   return df


def load_issues_fast(input_file):
   """
   Function for loading raw file in Pandas and formatting them that uses
   pandas builint function for faster processing
   """
   df = pd.read_csv(input_file, encoding="utf-8-sig", dtype="str", na_filter=False)
   # Fix problem with special characters
   # df = df.rename(columns={'\xef\xbb\xbfCASE_NUMBER': 'CASE_NUMBER'})
   # Fix problem with special characters
   # df = df.rename(columns={'\ufeffCASE_NUMBER': 'CASE_NUMBER'})
   # Remove dupliates
   df = df.drop_duplicates()
   # Format columns into correct type
#    df['CASE_NUMBER'] = df['CASE_NUMBER'].astype(str)
#    df['DETAILED_DESCRIPTION'] = df['DETAILED_DESCRIPTION'].astype(str)
#    df['ISSUE_SUMMARY'] = df['ISSUE_SUMMARY'].astype(str)
#    df['ATTRIBUTE_DOCUMENT_HEADER'] = df['ATTRIBUTE_DOCUMENT_HEADER'].astype(str)
#    df['ISSUE_DETECTION_POINT'] = df['ISSUE_DETECTION_POINT'].astype(str)
#    df['SOURCE_SYSTEM'] = df['SOURCE_SYSTEM'].astype(str)
#    df['CASE_STATUS'] = df['CASE_STATUS'].astype(str)
   # Creating pandas index
   df['idx'] = df['CASE_NUMBER']
   df = df.set_index('idx')
   # idx = df['CASE_NUMBER'].tolist() # no sure if this is needed
   df['DETAILED_DESCRIPTION'] = df['DETAILED_DESCRIPTION'].str.replace("\x96", "")
   #df['ISSUE_SUMMARY'] = df['ISSUE_SUMMARY'].str.replace("\x96", "")
   #df['SOURCE_SYSTEM'] = df['SOURCE_SYSTEM'].str.replace("\x96", "")
   return df


# nltk.data.find('.')
# choose english for stemming


def stem(text):
   """
   This function gets a text and first tokenize by sentence,
   then by word to ensure that punctuation is caught as it's own token
   """
   stemmer = SnowballStemmer("english")
   tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
   filtered_tokens = []
   # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
   for token in tokens:
       if search('[a-zA-Z]', token):
       # or search('[0-9]', token)
           filtered_tokens.append(token)
   stems = [stemmer.stem(t) for t in filtered_tokens]
   return stems


def make_tfidf(df):
   # TODO Write function description
   # df["sentence"] = df['DETAILED_DESCRIPTION'].map(str)+' '+df['ISSUE_SUMMARY'].map(str)+' '+df['ATTRIBUTE_DOCUMENT_HEADER'].map(str)+' '+df['SOURCE_SYSTEM'].map(str)+' '+df['ISSUE_DETECTION_POINT'].map(str)
   # sentence = df['sentence'].tolist()
   sentence = df['DETAILED_DESCRIPTION'].tolist()
   # + ' ' + df['ISSUE_SUMMARY'] + ' ' +\
    #          df['ATTRIBUTE_DOCUMENT_HEADER'] + ' ' + df['SOURCE_SYSTEM'] +\
        #      ' ' + df['ISSUE_DETECTION_POINT'].tolist()
   idx = df['CASE_NUMBER'].tolist()
   # create the Tfidf vectorizer function using some initial arguments
   tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english',
                                      tokenizer=stem)
   # fit (use sentence to create vocab) and transform (use sentence to find
   # idf) and then produce tdifd
   tfidf_matrix = tfidf_vectorizer.fit_transform(sentence)
   # save tfidf in a dataframe with columns of terms and rows of issues
   # the todense function change the sparse tfdif matrix to full matrix
   # toarray does the same thing
   df_tfidf_matrix = pd.DataFrame(data=tfidf_matrix.todense(),
                                  columns=tfidf_vectorizer.get_feature_names(),#sorted_x,
                                  index=idx)
   df_tfidf_matrix['CASE_NUMBER'] = idx # redundant
   return df_tfidf_matrix

# if __name__ == '__main__':
