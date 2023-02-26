import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from flask import Flask, request, render_template
import jyserver.Flask as jsf
from statistics import mode

df=pd.read_csv("static/Prototype.csv")
e= preprocessing.LabelEncoder()
df['prognosis']= e.fit_transform(df['prognosis'])
x=df.drop(["prognosis"],axis=1)
y=df["prognosis"]
temp=[0 for k in range(0,len(x.columns))]

#training data
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.2, random_state =42)

RF= RandomForestClassifier(random_state=42)
RF.fit(x_train, y_train)
y_pred1=RF.predict(x_test)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x, y)

SVM_=SVC()
SVM_.fit(x_train,y_train)
y_pred2=SVM_.predict(x_test)

svm_model=SVC()
svm_model.fit(x,y)

GB=GaussianNB()
GB.fit(x_train,y_train)
y_pred3=GB.predict(x_test)

gb_model=GaussianNB()
gb_model.fit(x,y)
#Testing data
t_df=pd.read_csv("static/Prototype-1.csv")
t_x=df.drop(["prognosis"],axis=1)
t_y=e.fit_transform(t_df['prognosis'])
t_y_pred=RF.predict(t_x)

d={"Diease":e.classes_}

data_des=pd.read_csv("static/symptom_Description.csv")
data_pre=pd.read_csv("static/symptom_precaution.csv")
app = Flask(__name__)
@app.route('/', methods =["GET", "POST"])
def predict():
	if(request.method=="POST"):
		l=[]
		s1=request.form.get("s1")
		s2=request.form.get('s2')
		s3=request.form.get('s3')
		l.extend([s1,s2,s3])
		print(l)
		for i in range(0,len(x.columns)):
			for j in l:
				if j==x.columns[i]:
					temp[i]=1
		input_data=np.array(temp).reshape(1,-1)
		p1=rf_model.predict(input_data)[0]
		p2=svm_model.predict(input_data)[0]
		p3=gb_model.predict(input_data)[0]
		result=mode([d["Diease"][p1],d["Diease"][p2],d["Diease"][p3]])
		d_prediction={ "SVM_model":d["Diease"][p1],"NB":d["Diease"][p2],"RB":d["Diease"][p3]}
		print(d_prediction)
		print(result)
		l1=list( data_des["Disease"])
		l2=list(data_des["Description"])
		z=list(data_pre["Disease"])
		for i in range(0,len(l1)):
			if l1[i]== result:
				k=l2[i]
				break
		for j in range(0,len(z)):
			if z[j]== result:
				z1=list(data_pre["Precaution_1"])[j]
				z2=list(data_pre["Precaution_2"])[j]
				z3=list(data_pre["Precaution_3"])[j]
				z4=list(data_pre["Precaution_4"])[j]
				break
		return render_template('result.html',r=result,K=k,c1=z1,c2=z2,c3=z3,c4=z4)
	return render_template('index.html')
if __name__=='__main__':
	app.run()
