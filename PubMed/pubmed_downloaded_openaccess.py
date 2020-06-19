import pandas as pd

# find out how many of the articles returned by PubMed search function that are available in PubMed's FTP service. 

articles = pd.read_csv('articles.csv')
index = pd.read_csv('oa_file_list.csv')

# include both commerical and non commerical subsets
pd.merge(articles[['PMID', 'TITLE']], index, how='inner', on='PMID').drop_duplicates(subset='PMID').to_csv('OpenAccess.csv')

print('articles.csv', pd.read_csv('articles.csv').shape)
print('OpenAccess.csv', pd.read_csv('OpenAccess.csv').shape)