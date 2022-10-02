# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 23:20:38 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%

leadtemplate2csv = pd.read_csv("wease_data_modified.txt")    
 
# %%
# data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# %%
M = leadtemplate2csv[leadtemplate2csv.Satis_Oldu_Mu == "Evet"]
B = leadtemplate2csv[leadtemplate2csv.Satis_Oldu_Mu == "Hayır"]
# scatter plot
plt.scatter(M.Cinsiyet,M.Tedavi_Turu,color="red",label="kotu",alpha= 0.3)
plt.scatter(B.Cinsiyet,B.Tedavi_Turu,color="green",label="iyi",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# %%
leadtemplate2csv.Satis_Oldu_Mu = [1 if each == "Evet" else 0 for each in leadtemplate2csv.Satis_Oldu_Mu]
y = leadtemplate2csv.Satis_Oldu_Mu.values
x_data = leadtemplate2csv.drop(["Satis_Oldu_Mu"],axis=8)

# %%
# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
 # %% SVM
 
from sklearn.svm import SVC
 
svm = SVC(random_state = 1)
svm.fit(x_train,y_train)
 
 # %% test
print("print accuracy of svm algo: ",svm.score(x_test,y_test))    




























