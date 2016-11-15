import pandas as pd
import numpy as np
import pylab as P
import csv as csv 

# Get the raw data
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)
ids = test_df['PassengerId'].values #keep the Id for the prediction

# This function prepare/clean the data for the classifiers
def cleanData(df):

	# Add columns (new feature, change types, fill some null values)
	df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
	df['Embarked_Num'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
	df.loc[ df['Embarked'].isnull(), 'Embarked_Num' ] = 0
	df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
	df['AgeFill'] = df['Age']
	df['FamilySize'] = df['SibSp'] + df['Parch']
	df['Age*Class'] = df.AgeFill * df.Pclass

	median_ages = np.zeros((2,3))
	for i in range(0, 2):
		for j in range(0, 3):
			median_ages[i,j] = df[(df['Gender'] == i) & \
								  (df['Pclass'] == j+1)]['Age'].dropna().median()
			df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
			 'AgeFill'] = median_ages[i,j]

			 
	# Remove data with 'object/data' type and column with null value
	df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
	df = df.drop( ['PassengerId'], axis=1) 
	df = df.dropna()
	return df

# Get clean data
train_data = cleanData(train_df).values
test_data = cleanData(test_df).values

# training our classifiers (RandomForest)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0])
output = forest.predict(test_data).astype(int)

#Write our answer in csv file
predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
