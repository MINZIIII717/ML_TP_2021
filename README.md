# ML_TP_2021

Kaggle Website: https://www.kaggle.com/wenruliu/adult-income-dataset

#1 result of preprocessing 
![image](https://user-images.githubusercontent.com/81156796/140601684-074f080a-eef5-4dea-8dfd-65aac26ba486.png)

#2 result of encoding&scaling
  ex) ordinal encoding & standardization scaling
  ![image](https://user-images.githubusercontent.com/81156796/140602045-595b226a-ce92-46cf-b3b3-347e58df3cde.png)

#3 result of classification
  #3-1 Ordinal encoding & standardization scaling
  Best : Logistic Regression {'C': 1.0, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l1', 'random_state': 3, 'solver': 'saga'} -0.22446976715459205
  ![image](https://user-images.githubusercontent.com/81156796/140601737-7a2485e3-d542-40fb-b2ad-ae9a04088f72.png)
  
  #3-2 Ordinal encoding & MinMax scaling
  Best : Logistic Regression {'C': 1.0, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l1', 'random_state': 3, 'solver': 'saga'} -0.22446976715459205
  ![image](https://user-images.githubusercontent.com/81156796/140601772-cd671d42-7d24-4f18-add9-09ba85843e20.png)
  
  #3-3 Ordinal encoding & MaxAbs scaling
  Best : Logistic Regression {'C': 1.0, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l1', 'random_state': 3, 'solver': 'saga'} -0.22446976715459205
  ![image](https://user-images.githubusercontent.com/81156796/140601805-a1b53a1d-94fa-47e3-93c7-587f4a79f3fa.png)
  
  #3-4 Ordinal encoding & Robust scaling
  Best : Logistic Regression {'C': 1.0, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l1', 'random_state': 3, 'solver': 'saga'} -0.22446976715459205
  ![image](https://user-images.githubusercontent.com/81156796/140601827-5067ab83-f3eb-4a57-855e-fc9a72c83a98.png)
  
  #3-5 OneHot encoding & standardization scaling
  Best : Logistic Regression {'C': 1.0, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l1', 'random_state': 3, 'solver': 'saga'} -0.22446976715459205
  ![image](https://user-images.githubusercontent.com/81156796/140601959-35786986-8289-463f-92e7-b4e752d6900b.png)
  
  #3-6 OneHot encoding & MinMax scaling
  Best : Logistic Regression {'C': 1.0, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l1', 'random_state': 3, 'solver': 'saga'} -0.22446976715459205
  ![image](https://user-images.githubusercontent.com/81156796/140601948-09f753c5-35f1-45fe-bc3e-a92ecff78467.png)
  
  #3-7 OneHot encoding & MaxAbs scaling
  Best : Decisioin Tree Regressor {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 5, 'splitter': 'best'} -0.22324606262738556
  ![image](https://user-images.githubusercontent.com/81156796/140601924-75e4be50-1810-4af0-8f2f-8b1b7d5a3ebf.png)

  #3-6 OneHot encoding & Robust scaling
  Best : Logistic Regression {'C': 1.0, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l1', 'random_state': 3, 'solver': 'saga'} -0.22446976715459205
  ![image](https://user-images.githubusercontent.com/81156796/140601904-33be010a-1432-4e17-bfc2-8a2b0603ee83.png)
  
  **** Best Result ****
  OneHot encoding & MaxAbs scaling with Decision Tree Regressor Accuracy : -0.22324606262738556
  parameter {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 5, 'splitter': 'best'}






