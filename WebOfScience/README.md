These scripts were built to compile data from Web of Science and download the PDF's using Kopernio.

# Data collection for the keywords heatmap
Given the list of keywords this scraper runs through the Web of Science Advanced Search with the possible 
keyword combinations and stores the number of results for the search. The final results are stored in 'wos_heatmap_data.csv'.

Instructions to run: 
* Include the keywords to be searched in 'keywords_A.csv' and 'keywords_B.csv' with the column header being 'keywords' in the same folder as the python script.
* Install the packages listed in wos_heatmap_scraper_requirements.txt.
* Run python wos_heatmap_scraper.py.


