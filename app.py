import flask
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings; warnings.simplefilter('ignore')
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Use pickle to load in the pre-trained model.
def tune_model(input_variables, no_of_trees):
	print("=====No of estimator===")
	print(no_of_trees)
	if no_of_trees == 20:
		with open(f'model/random-forest-classifier.pkl', 'rb') as f:
			rf_model = pickle.load(f)
	else:
		print("---Tune Model----")
		bankdata = pd.read_csv("bill_authentication.csv")
		X = bankdata.drop('Class', axis=1)  
		y = bankdata['Class']		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)		
		classfier = RandomForestClassifier(n_estimators=20, random_state= 0)
		classfier.fit(X_train, y_train)
		y_pred = classfier.predict(X_test)
		with open('model/random-forest-classifier1.pkl', 'wb') as file:
			pickle.dump(classfier, file)
		with open(f'model/random-forest-classifier1.pkl', 'rb') as f:
			rf_model = pickle.load(f)

	prediction =rf_model.predict(input_variables)[0]
	print("-----Prediction------")
	print(prediction)
	return prediction


with open(f'model/svm-classifier.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open(f'model/decision-tree-classifier.pkl', 'rb') as f:
    dt_model = pickle.load(f)



app = flask.Flask(__name__, template_folder='templates')
app.debug = True
@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))

	if flask.request.method == 'POST':

		model_choice = flask.request.form['model']
		#print(model_choice)
		variance = flask.request.form['variance']
		skewness = flask.request.form['skewness']
		curtosis = flask.request.form['curtosis']
		entropy  = flask.request.form['entropy']
		estimator = flask.request.form['estimator']

		print(f"--Estimaor---{estimator}")

		input_variables = pd.DataFrame([[variance, skewness, curtosis, entropy]], columns=['variance', 'skewness', 'curtosis' ,'entropy'], dtype=float)

		#print(input_variables)

		if model_choice == svm_model:
			prediction = svm_model.predict(input_variables)[0]
		elif model_choice == dt_model:
			prediction = dt_model.predict(input_variables)[0]
		else:
			#prediction = rf_model.predict(input_variables)[0]
			prediction = tune_model(input_variables, estimator)

		print(f"Prediction-{prediction}")
		return flask.render_template('main.html', result='True', original_input={
			'variance' : variance,
			'skewness' : skewness,
			'curtosis' : curtosis,
			'entropy'  : entropy
			},
			Prediction= prediction)


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/build', methods=['GET', 'POST'])
def build():
	if flask.request.method == "GET":
		msg = "Upload a file in csv format only"		

	if flask.request.method == "POST":
		input_file = flask.request.files['input_file']
		print(input_file)
		if input_file:
			df = pd.read_csv(input_file)
			print(df.head())
			feature_names = ['OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore']
			training_features = df[feature_names]

			outcome_name = ['Recommend']
			outcome_labels = df[outcome_name]

			numeric_feature_names = ['ResearchScore', 'ProjectScore']
			categoricial_feature_names = ['OverallGrade', 'Obedient']

			ss = StandardScaler()
			# fit scaler on numeric features
			ss.fit(training_features[numeric_feature_names])

			# scale numeric features now
			training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])
			# view updated feature-set
			print(training_features)

			#--Engineering Categorical Features
			training_features = pd.get_dummies(training_features, columns=categoricial_feature_names)
			print(training_features)

			#--get list of new categorical features
			categorical_engineered_features = list(set(training_features.columns) - set(numeric_feature_names))
			print(categorical_engineered_features)

			X_train, X_test, y_train, y_test  = train_test_split(training_features, outcome_labels, test_size = 0.25)
			model = LogisticRegression()
			model.fit(X_train, y_train)

			#--simple evaluation on training data
			pred_labels = model.predict(training_features)
			actual_labels = np.array(outcome_labels['Recommend'])


			print('Accuracy:', float(accuracy_score(actual_labels, pred_labels))*100, '%')
			print('Classification Stats:')
			print(classification_report(actual_labels, pred_labels))

			Accuracy = float(accuracy_score(actual_labels, pred_labels))*100
			msg = "Accuracy of this model is" +str(Accuracy)+ "%"

			
	return flask.render_template('build.html', msg=msg)


@app.route("/test_model", methods=['GET','POST'])
def test_model():
	if flask.request.method == "GET":
		msg = "Please Input All required Valid Data"

	if flask.request.method == "POST":
		name = flask.request.form['name']
		overallgrade = flask.request.form['overallgrade']
		obedient = flask.request.form['obedient']
		projectscore = flask.request.form['projectscore']
		researchscore = flask.request.form['researchscore']

		
	return flask.render_template('test_model.html')

if __name__ == '__main__':
	app.run(debug=True)