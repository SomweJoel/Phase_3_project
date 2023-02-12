# Phase_3_project
CHICAGO CAR CRASHES PROJECT

INTRODUCTION

For this classification project, the aim is to use machine learning concepts to predicting whether or not a car crash is fatal.

OBJECTIVES

MAIN OBJECTIVE

To build a classification model to predict whether a car crash is fatal or not.

SPECIFIC OBJECTIVES

To clean and merge the three datasets(crashes,people,vehicles) so as to make it easy for modelling
Identify features in the data and use them for engineering a model
Evaluating our model
Come up with interpretations upon interpreting our classifier model

BUSINESS UNDERSTANDING

What is the problem?
To help car companys make their buyers feel safer by being able to identify which crashes are fatal or not so that emergency services can respond quickly to fatal cases and give them prority.

Farther understanding

The proposed software based on the model of this project is used to potentially save the life of a driver in the event of a car accident. The software takes record of the features occuring at the time of the accident and if they are highly correlated with those of previous fatal crashes, athorities are immediately notified of possible death.If successful and car companies implement our recommendations, we will see a decrease in fatalities and an overall increase in safety for all citizens.

DATA UNDERSTANDING

Overview

I used Chicago,IL car crash data from city of Chicago website the 3 data sets contain crash info, info on people involved and info on vehicles involved.

I used the 3 data sets below from the City of Chicago website and combined them.

Traffic Crashes - Crashes: https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if

Traffic Crashes - Vehicle: https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3

Traffic Crashes - People: https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d
Load Data

In order to use run this data you must download the data sets from the following 3 links and store them in the folder above your notebook

https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if
https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3
https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d

Data Cleaning

Merge all three data sets together and drop duplicates.
# Merging Datasets on the Crash_Record_ID Column
df_merge = pd.merge(df, df_vehicles, on='CRASH_RECORD_ID').reset_index()
df_merge_2 = pd.merge(df_merge, df_people, on='CRASH_RECORD_ID').reset_index()

#dropping dupllicates(checks CRASH_RECORD_ID for uniqueness)
df_dropped= df_merge_2.drop_duplicates(subset=['CRASH_RECORD_ID'], keep='first')

Check the distribution of the columns of the dataset.
![image](https://user-images.githubusercontent.com/117165965/218337457-3bc59b3b-c653-4005-ab20-38a68433f355.png)

Modeling

Dummy Classfier

Baseline model for comparision.

dummy = DummyClassifier(random_state=42) 

#establishing random_state for reproducibility
dummy.fit(X_train_clean, y_train)
y_pred = dummy.predict(X_test_clean)
cm = confusion_matrix(y_test,y_pred)

ConfusionMatrixDisplay(cm, display_labels=["True label", "Predicted label"]).plot()

![image](https://user-images.githubusercontent.com/117165965/218337734-27615913-1244-4ca1-8bd0-7b4f55deec6b.png)

SMOTE

We have a class imbalance so we're trying to oversample the minority class of our target.

# X_train_clean.columns
smote = SMOTE(sampling_strategy='auto',random_state=42)
X_train_resampled, y_train_resampled = smote.fit_sample(X_train_clean, y_train) 

Dummy Classfier with SMOTE

dummy_smote = DummyClassifier(random_state=42)
dummy_smote.fit(X_train_resampled, y_train_resampled)
y_pred_dummy_sm = dummy_smote.predict(X_test)
cm_2 = confusion_matrix(y_test,y_pred_dummy_sm)

ConfusionMatrixDisplay(cm_2, display_labels=["True label", "Predicted label"]).plot()

![image](https://user-images.githubusercontent.com/117165965/218337807-f7d5fae9-80af-4690-9773-688aa47fd2de.png)

Decision Tree

tree = DecisionTreeClassifier()
Doing a grid search to find the best parameters

tree_grid = {'max_leaf_nodes': [4, 5, 6, 7], 
             'min_samples_split': [2, 3, 4],
             'max_depth': [2, 3, 4, 5],
            }
            
plot_tree(best_tree)
plt.savefig('images/decision_tree.png');

![image](https://user-images.githubusercontent.com/117165965/218338105-a5f6f848-20a2-4e47-b483-e12afd2ef6f6.png)

Best tree confusion matrix

![image](https://user-images.githubusercontent.com/117165965/218338134-dd166520-c7bb-4720-bd9d-cddf5bfd31ec.png)

Best tree ROC curve

![image](https://user-images.githubusercontent.com/117165965/218338174-b7cebbe5-03d5-4e7f-b641-be82ad96f60b.png)

Checking how much accuracy the model loses by excluding each variable

![image](https://user-images.githubusercontent.com/117165965/218338197-8837a279-da0a-4b34-af22-8c493f840f97.png)

Random Forest

![image](https://user-images.githubusercontent.com/117165965/218338218-98267c9c-f192-4745-832d-19825e47f8fe.png)

Final Model Evaluation

Decision Tree:

Accuracy: 0.9180
AUC: 0.79
Precision: 0.0079
Recall: 0.4884
F1 Score: 0.0155

This final model is excellent at predicting whether or not a car crash is fatal. Therefore, this model should be used by car companies as a safety feature in their vehicles to help 911 dispatchers determine whether or not certain crashes are a priority or not.

Reccomendations

The best model for use is the decision tree because it has a relatively high accuracy of 91.8% meaning it is correct 91.8% of the time, however, the precision of 0.0079 indicates that, out of all the positive predictions made by the model, only 0.79% of them were actually positive. This is a very low precision value, which means that the model is making a large number of false positive predictions.This leads to the following reccomendations:

Consider using different feature selection or engineering techniques to improve the performance of the model.
Consider adjusting the parameters of the Decision Tree model to optimize its performance.

Conclusion

In conclusion, the Decision Tree model has a relatively high accuracy, but is not performing well in terms of precision and recall. Further work may be necessary to improve the performance of the model and achieve better results. Still, it would be beneficial to use this model as a safety feature because even though precision is low,better safe than sorry.
