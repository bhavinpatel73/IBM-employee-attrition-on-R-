# IBM-employee-attrition-on-R-
In this project, I tried to analyze what factors lead to employee retention in the company, and what factors influence them the most with the help of some EDA analysis and some machine learning algorithms. All this has been done with the help of RStudio and Tableau. Dataset had 1470 observations with 35 variables in total. Dataset had 1470 observations with 35 variables in total. By having a look at the dataset, I concluded that there where 4 variables which had the same values throughout the dataset(EmplyeeCount, EmployeeNumber, Over18, StandardHours). After dropping these variables I went on with the EDA and modeling part. When I took the variable wise analysis, it was concluded that Overtime, MartialStatus, Department, Business Travel, etc where factors which were affecting the attrition rate. Also plotting some variables in boxplot I found outliers on Monthly Income, Total Working Hours, Years at Company, etc. In the modeling part, I had included Logistic regression, SVM, Random Forest, Neural Networks and XGBoost. The best model with good accuracy and less error turn up to be XGBoost with 88.58 %.
