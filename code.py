# --------------
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import sys
import csv



#from io import StringIO
#with StringIO(file_content) as f:
#    data = np.loadtxt(f, skiprows=1, dtype=int)


#Code Starts here
# Command to display all the columns of a numpy array
np.set_printoptions(threshold=sys.maxsize)

X=[]
# Load the data. Data is already given to you in variable `path` 
with open(path,'r') as f:
  csv_data=csv.reader(f)
  next(csv_data)
  for row in csv_data:
    X.append(row)
data=np.asarray(X)
#print(data)
print('='*100)


from collections import Counter
# How many unique ad campaigns (xyz_campaign_id) does this data contain ? And for how many times was each campaign run ?
print('Number of Ad Campaigns according to ID (xyz_campaign_id)',Counter(np.apply_along_axis(lambda x: x[1],1,data)))
print('Unique Number of Ad Campaigns (xyz_campaign_id)',len(Counter(np.apply_along_axis(lambda x: x[1],1,data))))
print('='*100)

# What are the age groups that were targeted through these ad campaigns?
#print(Counter(np.apply_along_axis(lambda x: x[3],1,data)))
print('Age Groups Targeted by Ad Campaign',np.unique(np.apply_along_axis(lambda x: x[3],1,data)))
print('='*100)

# What was the average, minimum and maximum amount spent on the ads?
print('Average amount spent on Ad',np.apply_along_axis(lambda x: x[8].astype(float),1,data).mean())
print('Minimum amount spent on Ad',np.apply_along_axis(lambda x: x[8].astype(float),1,data).min())
print('Maximum amount spent on Ad',np.apply_along_axis(lambda x: x[8].astype(float),1,data).max())
print('='*100)

# What is the id of the ad having the maximum number of clicks ?
#print(data[np.where(np.apply_along_axis(lambda x: x[7].astype(int),1,data) ==np.max(np.apply_along_axis(lambda x: x[7].astype(int),1,data)))[0][0]][0])
print('ID of Ad with maximum number of clicks',data[np.argmax(np.apply_along_axis(lambda x: x[7].astype(int),1,data),axis=0)][0])
print('='*100)

# How many people bought the product after seeing the ad with most clicks? Is that the maximum number of purchases in this dataset?
print('Number of people who bought product for Ad with Most Clicks',data[np.argmax(np.apply_along_axis(lambda x: x[7].astype(int),1,data),axis=0)][10])
print('='*100)
print('Maximum number of purchases',np.apply_along_axis(lambda x: x[10].astype(int),1,data).max())
print('='*100)

# So the ad with the most clicks didn't fetch the maximum number of purchases. 
#Find the details of the product having maximum number of purchases
print('Details of Product with maximum number of purchases')
product_details=['ad_id','xyz_campaign_id','fb_campaign_id','Age','Gender','interest','Impressions','Clicks','Spent','Total_Conversion','Approved_Conversion']
c=0
for i in product_details:		
  print(i,data[np.argmax(np.apply_along_axis(lambda x: x[10].astype(int),1,data),axis=0)][c])
  c=c+1		
print('='*100)

# Create a new feature `Click Through Rate`  (CTR) and then concatenate it to the original numpy array 
#CTR= Clicks ×100 / Impressions
CTR=(np.apply_along_axis(lambda x: x[7].astype(float),1,data)*100/np.apply_along_axis(lambda x: x[6].astype(float),1,data))
data=np.concatenate((data,CTR.reshape(-1,1)),axis=1)
display(data.shape)

# Create a new column that represents Cost Per Mille (CPM) 
CPM=(np.apply_along_axis(lambda x: x[8].astype(float),1,data)*100/np.apply_along_axis(lambda x: x[6].astype(float),1,data))
data=np.concatenate((data,CPM.reshape(-1,1)),axis=1)
display(data.shape)


