import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


df = pd.read_csv('laptop_data.csv')

print(df.columns)

df = df[['Company', 'TypeName', 'Inches', 'ScreenResolution',
       'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price']]

catvars = df.select_dtypes(include=['object']).columns
numvars = df.select_dtypes(include = ['int32','int64','float32','float64']).columns


def uniquevals(col):
    print(f'Details of the particular col {col} is : {df[col].unique()}')
    
    
def valuecounts(col):
    print(f'Valuecounts of the particular col {col} is : {df[col].value_counts()}')
    
    
for col in df.columns:
    uniquevals(col)
    print("-"*75)


'''
so on observation we can see that if we remove "GB" from RAM,i can 
make it as an integer value then after,now same goes with Memory as 
well as Weight,for Weight i can classify it as floating variable
using the str.replace() as shown ↓
'''

df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

# converting from string->integer for ram column

df['Ram'] = df['Ram'].astype('int32')

# converting from string-> float for the weight column

df['Weight'] = df['Weight'].astype('float32')


# **For the `Screen Resolution` column we have many types of Screen Resolutions out there as shown `Touch Screen` and `Normal` and `IPS Panel` are the 3 parts on basis of which we can segregate the things**

# creating a new col,touchscreen if the value is 1 that laptop is touch screen

df['TouchScreen'] = df['ScreenResolution'].apply(lambda element:1 
                                                      if 'Touchscreen' in element else 0)
# creating a new col named IPS,does the laptop have IPS facility or not

df['IPS'] = df['ScreenResolution'].apply(
    lambda element:1 if "IPS" in element else 0
)


# ### Extracting the X Resolution and the Y Resolution

# we will split the text at the "x" letter and seperate the 2 parts
# from this we can observe that one of the col is Y res we need to do
# some feature engineering on the X res col

splitdf = df['ScreenResolution'].str.split('x',n = 1,expand=True)

splitdf = df['ScreenResolution'].str.split('x',n = 1,expand=True)

df['X_res'] = splitdf[0]
df['Y_res'] = splitdf[1]


'''
So basically from that whole text of the X_res col,we need to 
extract the digits from it,but the problem is the numbers are scattered 
in some cases,that is the reason why i am using regex,if we use this
we will exactly get the numbers which we are looking for!,
so firstly replace all the "," with "" and then find all numbers
from that string as "\d+\.?\d+",\d means that integer number and \.? 
all the numbers which come after an number and \d+ the string must end with number
'''


df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])



# In[24]:


df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')


plt.figure(figsize=(15,7))
sn.heatmap(df.corr(),annot=True,cmap='plasma')
plt.show()

# **From the correlation plot we observed that as the X_res and Y_res is increasing,the price of the laptop is also increasing,so `X_res and Y_res` are positively correlated and they are giving much information,so that is the reason why i had splitted `Resolution` column into `X_res and Y_res` columns respectively**

# **So to make things good,we can create a new column named `PPI{pixels per inch}`,now  as we saw from the correlation plot that the `X_res and Y_res` are having much collinearity,so why not combine them with `Inches` which is having less collinearity,so we will combine them as follows ↓,so here is the formula of how to calculate `PPI` {pixels per inch}**

# $$
#     PPI(pixels per inch) = \frac{\sqrt{X_resolution^2+Y_resolution^2}}{inches}
# $$

df['PPI'] = (((df['X_res']**2+df['Y_res']**2))**0.5/df['Inches']).astype('float')
df.head()


# **So as we observe from the correlation data that the `PPI` is having good correlation,so we will be using that,as that is a combination of 3 features and that gives collective results of 3 columns,so we will drop `Inches,X_res,Y_res` as well**

df.drop(columns=['ScreenResolution','Inches','X_res','Y_res'],inplace=True)
df.head()


# **Now we will work on `CPU` column,as that also has much text data and we need to process it efficiently as we may get good insights from them**

print(df['Cpu'].value_counts())


# **Most common processors are made by intel right,so we will be clustering their `processors` into different categories like `i5,i7,other`,now other means the processors of intel which do not have i3,i5 or i7 attached to it,they're completely different so that's the reason i will clutter them into `other` and other category is `AMD` which is a different category in whole**
# 
# **So if we observe we need to extract the first 3 words of the CPU column,as the first 3 words of every row under the CPU col is the type of the CPU,so we will be using them as shown ↓**

df['CPU_name'] = df['Cpu'].apply(lambda text:" ".join(text.split()[:3]))

'''
As mentioned earlier,if we get any of the intel `i3,i5 or i7` versions
we will return them as it is,but if we get any other processor
we will first check whether is that a variant of the intel? or not
if yes,then we will tag it as "Other Intel Processor" else we will
say it as `AMD Processor`

'''

def processortype(text):
    
    if text=='Intel Core i7' or text=='Intel Core i5' or text=='Intel Core i3':
        return text
    
    else:
        if text.split()[0]=='Intel':
            return 'Other Intel Processor'
        
        else:
            return 'AMD Processor'
        
    
    
df['CPU_name'] = df['CPU_name'].apply(lambda text:processortype(text))

## dropping the cpu column

df.drop(columns=['Cpu'],inplace=True)


# ##### Analysis on the RAM column

# ##### About the memory column

# **We will seperate the `Type` of memory and the value of it,just similar to the one which is done in the previous part**
# 
# **This part involves things which are needed to be done in steps,so here we do not have the memory as a complete we have it in different dimension as `128GB SSD +  1TB HDD`,so inorder to for it come in a same dimension we need to do some modifications which are done below as shown**


## 4 most common variants observed : HHD,SSD,Flash,Hybrid

# this expression will remove the decimal space for example 1.0 TB will be 1TB

df['Memory'] = df['Memory'].astype(str).replace('\.0','',regex = True)

# replace the GB word with " "

df['Memory'] = df['Memory'].str.replace('GB','')

# replace the TB word with "000"

df['Memory'] = df['Memory'].str.replace('TB','000')

# split the word accross the "+" character

newdf = df['Memory'].str.split("+",n = 1,expand = True)

# we will strip up all the white spaces,basically eliminating white space

df['first'] = newdf[0]
df['first'] = df['first'].str.strip()


def applychanges(value):
    
    df['Layer1'+value] = df['first'].apply(lambda x:1 if value in x else 0)
    
    
listtoapply = ['HDD','SSD','Hybrid','FlashStorage']    
for value in listtoapply:
    applychanges(value)
    

# remove all the characters just keep the numbers

df['first'] = df['first'].str.replace(r'\D','')

df['Second'] = newdf[1]


def applychanges1(value):
    
    df['Layer2'+value] = df['Second'].apply(lambda x:1 if value in x else 0)
    
    
listtoapply1 = ['HDD','SSD','Hybrid','FlashStorage']
df['Second'] = df['Second'].fillna("0")
for value in listtoapply1:
    applychanges1(value)
    

# remove all the characters just keep the numbers

df['Second'] = df['Second'].str.replace(r'\D','')
df['first'] = df['first'].astype('int')
df['Second'] = df['Second'].astype('int')

# multiplying the elements and storing the result in subsequent columns


df["HDD"]=(df["first"]*df["Layer1HDD"]+df["Second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["Second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["Second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1FlashStorage"]+df["Second"]*df["Layer2FlashStorage"])


## dropping of uncessary columns

df.drop(columns=['first', 'Second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1FlashStorage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2FlashStorage'],inplace=True)


df.drop(columns=['Memory'],inplace=True)

# **Based on the correlation we observe that `Hybrid` and `Flash Storage` are almost negligible,so we can simply drop them off,where as HDD and SDD are having good correlation,we find that HDD has -ve relation with Price,and that's true,if the price of laptop is increasing there is more probability that the laptop is gonna use SDD instead of HDD and vice versa as well**


df.drop(columns = ['Hybrid','Flash_Storage'],inplace=True)


# ##### Analysis on GPU


df['Gpu'].value_counts()


# **Here as we are having less data regarding the laptops,its better that we focus on `GPU brands` instead focusing on the values which are present there beside them,we will focus on the `brands`**
# this is what we will be doing,extracting the brands 
a = df['Gpu'].iloc[1]
print(a.split()[0])


# In[51]:


df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
sn.countplot(x='Gpu brand',data=df,palette='plasma')

# removing the "ARM" tuple

df = df[df['Gpu brand']!='ARM']

# price-GPU analysis,i used np.median inorder to check if there is any
# inpact of outlier or not

sn.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)

df = df.drop(columns=['Gpu'])
df.head()


# ##### Operating System analysis
df['OpSys'].value_counts()

df['OpSys'].unique()

# club {Windows 10,Windows 7,Windows 7 S}-->Windows
# club {macOS,mac OS X}--> mac
# else return Others

def setcategory(text):
    
    if text=='Windows 10' or text=='Windows 7' or text=='Windows 10 S':
        return 'Windows'
    
    elif text=='Mac OS X' or text=='macOS':
        return 'Mac'
    
    else:
        return 'Other'
    
    
df['OpSys'] = df['OpSys'].apply(lambda x:setcategory(x))

# Price Analysis

# correlation with price

plt.figure(figsize=(10,5))
sn.heatmap(df.corr(),annot=True,cmap='plasma')
plt.show()

# ## Model Building

import numpy as np
test = np.log(df['Price'])
train = df.drop(['Price'],axis = 1)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(train,test,
                                                   test_size=0.15,random_state=2)

X_train.shape,X_test.shape


# **There's a Class which we imported named as `Column Trasnformer` we use this widely while building our models using `Pipelines`,so for this we have to get the index numbers of the columns which are having categorical variables**

mapper = {i:value for i,value in enumerate(X_train.columns)}
mapper


# ### Linear Regression

# we will apply one hot encoding on the columns with this indices-->[0,1,3,8,11]
# the remainder we keep as passthrough i.e no other col must get effected 
# except the ones undergoing the transformation!

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,3,8,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',metrics.r2_score(y_test,y_pred))
print('MAE',metrics.mean_absolute_error(y_test,y_pred))

## now mae is 0.21 so if you want to check how much difference is there do this

## we see there is a difference of 1.23 only as per the orignal value
## that is our model predicts +-0.21 more/less than the original price!

# ### Ridge Regression
# we will apply one hot encoding on the columns with this indices-->[0,1,3,8,11]
# the remainder we keep as passthrough i.e no other col must get effected 
# except the ones undergoing the transformation!

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,3,8,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',metrics.r2_score(y_test,y_pred))
print('MAE',metrics.mean_absolute_error(y_test,y_pred))


# ### LassoRegression
# we will apply one hot encoding on the columns with this indices-->[0,1,3,8,11]
# the remainder we keep as passthrough i.e no other col must get effected 
# except the ones undergoing the transformation!

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,3,8,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',metrics.r2_score(y_test,y_pred))
print('MAE',metrics.mean_absolute_error(y_test,y_pred))


# ### Decision Tree
# we will apply one hot encoding on the columns with this indices-->[0,1,3,8,11]
# the remainder we keep as passthrough i.e no other col must get effected 
# except the ones undergoing the transformation!

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,3,8,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',metrics.r2_score(y_test,y_pred))
print('MAE',metrics.mean_absolute_error(y_test,y_pred))


# ### Random Forest
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,3,8,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',metrics.r2_score(y_test,y_pred))
print('MAE',metrics.mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,3,8,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

data = pd.read_csv("traineddata.csv")

# print("Available options for Company:", data['Company'].unique())
# print("Available options for Type:", data['TypeName'].unique())
# print("Available options for Ram(in GB):", [2, 4, 6, 8, 12, 16, 24, 32, 64])
# print("Available options for OS:", data['OpSys'].unique())
# print("Enter weight of the laptop:")
# print("Available options for Touchscreen: No, Yes")
# print("Available options for IPS: No, Yes")
# print("Enter screen size:")
# print("Available options for Screen Resolution: 1920x1080, 1366x768, 1600x900, 3840x2160, 3200x1800, 2880x1800, 2560x1600, 2560x1440, 2304x1440")
# print("Available options for CPU:", data['CPU_name'].unique())
# print("Available options for HDD(in GB):", [0, 128, 256, 512, 1024, 2048])
# print("Available options for SSD(in GB):", [0, 8, 128, 256, 512, 1024])
# print("Available options for GPU(in GB):", data['Gpu brand'].unique())

# # Get input values from user
# company = input("Enter the Brand of the laptop: ")
# type = input("Enter the Type of the laptop: ")
# ram = int(input("Enter the RAM of the laptop (in GB): "))
# os = input("Enter the OS of the laptop: ")
# weight = (input("Enter the Weight of the laptop: "))
# touchscreen = input("Does the laptop have a touchscreen? (Enter Yes or No): ")
# ips = input("Does the laptop have IPS? (Enter Yes or No): ")
# screen_size = float(input("Enter the Screen Size of the laptop: "))
# resolution = input("Enter the Screen Resolution of the laptop: ")
# cpu = input("Enter the CPU of the laptop: ")
# hdd = int(input("Enter the HDD size of the laptop (in GB): "))
# ssd = int(input("Enter the SSD size of the laptop (in GB): "))
# gpu = input("Enter the GPU of the laptop: ")



company = "HP"
type = "Gaming"
ram = 8
os = "Windows"
weight = 1.8
touchscreen = 'No'
ips = 'No'
screen_size = 15.6
resolution = "1920x1080"
cpu = "Intel Core i5"
hdd = 0
ssd = 512
gpu = "Intel"


# Convert input values to appropriate format for prediction
ppi = None
if touchscreen == 'Yes':
    touchscreen = 1
else:
    touchscreen = 0

if ips == 'Yes':
    ips = 1
else:
    ips = 0
    
X_resolution = int(resolution.split('x')[0])
Y_resolution = int(resolution.split('x')[1])
ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

dict_for_df ={
        'Company': company,
         'TypeName' : type,
         'Ram' : ram,
         'OpSys' : os,
         'Weight' : weight,
          'TouchScreen' : touchscreen,
           'IPS' : ips,
          'PPI' : ppi,
          'CPU_name' : cpu,
          'HDD' : hdd,
           'SSD' : ssd,
          'Gpu brand' : gpu
    }
# 
    # Now convert dictionary into dataframe.
df= pd.DataFrame(dict_for_df,index=[0])
    
#y_pred = pipe.predict(df)


y_pred = str(int(np.exp(pipe.predict(df)))) 



print("Your Laptop price is ₹ {0}/-".format(y_pred))




print("---------------------------------------------------------------------")

# Web Scraping code to display top 5 laptops with price in the predicted range
import requests
from bs4 import BeautifulSoup

lower_bound = int(y_pred) - 5000
upper_bound = int(y_pred) + 5000

url = "https://www.amazon.in/s?k=" + company + "+laptop+" + str(ram) + "+gb+ram+" + gpu.replace(' ', '+') + "+gpu+" + cpu.replace(' ', '+') + "&rh=n%3A1375424031&dc&qid=1620509431&rnid=1375425031&ref=sr_nr_n_1"

response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
results = soup.find_all("div", {"class": "s-result-item"})

count = 0
for result in results:
    try:
        name = result.find("span", {"class": "a-size-medium a-color-base a-text-normal"}).get_text().strip()
        price = result.find("span", {"class": "a-offscreen"}).get_text().strip()
        price = int(price.replace(",", "").replace("₹", ""))
        if price >= lower_bound and price <= upper_bound:
            print(name)
            print("Price: ₹", price)
            print()
            count += 1
            if count == 5:
                break
    except:
        pass

print("---------------------------------------------------------------------")


