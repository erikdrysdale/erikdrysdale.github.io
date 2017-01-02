# Python code for munging data for the Batch effects paper

import numpy as np
import pandas as pd

# Set up directory
import os
dir = 'C:/Users/erikinwest/Documents/Courses/EPID_823/presentation/blog'
os.chdir(dir)
os.listdir(os.getcwd())
# Define the directory where we keep the raw CEL files
raw_dir = 'C:/Users/erikinwest/Documents/Courses/EPID_823/presentation/raw'

####################################################################
############ ----- PART 1: GET BM NAMES FROM NCBI ----- ############
####################################################################

# Load in webscraping modules
from lxml import html
import requests

# Get the CEL File names
raw_names = np.asarray(os.listdir(raw_dir))
cel_match = np.asarray([x.find('CEL.gz') for x in raw_names])
# Now get the names we have CEL files for
cel_names = raw_names[np.where(cel_match>0)]
# Remove any .CEL.gz
cel_names = np.asarray([x.replace('.CEL.gz','') for x in cel_names])

# Define base NCBI site
ncbi_site = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
# Define the XPath
ncbi_xpath = '//td//td//tr[(((count(preceding-sibling::*) + 1) = 3) and parent::*)]//td/text()'

# Create storage
bm = np.array(range(len(cel_names)),dtype='<U9')

# Loop over all CEL files
for x in range(len(cel_names)):
    # Load the page
    temp_ncbi = ncbi_site + cel_names[x]
    temp_page = requests.get(temp_ncbi)
    temp_tree = html.fromstring(temp_page.content)
    # Pull the XPath
    temp_bm = temp_tree.xpath(ncbi_xpath)
    # Remove the underscore, if any
    temp_bm = temp_bm[1].split('_')[0]
    # Store
    bm[x] = temp_bm
    print(x,temp_bm)
    del temp_ncbi, temp_page, temp_tree, temp_bm

bm
    
####################################################################
############ ----- PART 2: GET BM NAMES FROM NCBI ----- ############
####################################################################

# Define base Coriell site
coriell_site = "https://catalog.coriell.org/0/Sections/Search/Sample_Detail.aspx?Ref="

# Define the relevant XPATH
coriell_xpath = '//span[(((count(preceding-sibling::*) + 1) = 1) and parent::*)]/text()'

# Create storage for ethnic labels
eth = list()

# Loop over the bm codes
for x in range(len(bm)):
    # Load the page
    temp_coriell = coriell_site + bm[x]
    temp_page = requests.get(temp_coriell)
    temp_tree = html.fromstring(temp_page.content)
    # Pull the XPath
    temp_eth = temp_tree.xpath(coriell_xpath)
    # Store
    eth.append(temp_eth)
    print(x,temp_eth)
    del temp_coriell, temp_page, temp_tree
    
# Find if any data is missing
missing_eth=np.asarray([len(x) for x in eth])
np.where(missing_eth==9)
# Manually checking the NCBI website confirms that they are Caucasian

eth2 = np.array(range(len(bm)),dtype='<U9')

for k in range(len(eth)):
    if len(eth[k])<10:
        eth2[k] = 'Caucasian'
    elif eth[k][10]=='Epstein-Barr Virus':
        eth2[k] = eth[k][11]
    else:
        eth2[k] = eth[k][10]

eth2

# Save data in a pandas data.frame from R analysis
eth_dat = pd.DataFrame({'eth':eth2 ,'code':bm,'gsm':cel_names})
eth_dat.to_csv('python_eth.csv',Index=None)


##################################################################
############ ----- PART 3: GET THE ACROBAT DATA ----- ############
##################################################################

# Load in the column header to begin building the data frame
dat = pd.read_csv('supp1.txt',sep=' ',nrows=1,header=None,skiprows=range(0))
# Get the "right" number of columns
p = dat.size

# Loop over the supp1.txt file to built a numpy!
j = 0
while(j>=0):
    j = j + 1
    try:
        # Try to load in the data
        row = pd.read_csv('supp1.txt',sep=' ',nrows=1,header=None,skiprows=range(j))
        # Update the data if we match the right length
        if row.size==p:
            dat = pd.concat([dat,row])
        else:
            print("Something else for j = ",j)
    except:
        print("we failed on j =",j)
        j = -1

# Set the right index
dat.index = np.arange(0,len(dat))

# Save the data for later
dat.to_csv('supp1.csv',header=None,index=None)





