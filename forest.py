import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#dataset
data=pd.read_csv('covertype.csv')
# print(data)
#train and test data
X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']
#splitting
Xtrain,Xtest,yTrain,yTest=train_test_split(X,y,test_size=0.2)
#scaling
st=StandardScaler()
Xtrain=st.fit_transform(Xtrain)
Xtest=st.fit_transform(Xtest)

# training
rand=RandomForestClassifier()
rand.fit(Xtrain,yTrain)

pred=rand.predict(Xtest)

print("Accuracy:", rand.score(Xtest, yTest))

cm = confusion_matrix(yTest, pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
