# Link to dataset
https://www.kaggle.com/datasets/hetbabariya/imdb-movies-data-collection-5000-records/data
# Project - Create your own predictive model

Predictive models are used across industries to analyze and make predictions about data. From sports to beauty products to app usage, predictive models provide individuals and businesses with data to make informed decisions.

Throughout this module, you have learned how to program both supervised and unsupervised learning models. In this final project, you will create your own! Throughout this project, you will work through activities to complete the following steps:

1. Choose a dataset.
2. Determine which algorithm is the best fit. Based on the dataset you choose, you will need to figure out which algorithm to use. This will require you to get to know your data and your goals! Is there a linear correlation between variables? Are you looking for numerical value or a label/category? Do you know the labels or do you need the model to create them for you?
- Do some tests with matplotlib and visualize your data.  Does it provide a good correlation?  Why or why not?
3. Program your model. Once you have chosen your type of model, it’s time to create it! In this step, you will write a program that fits your chosen model to the data. Your program and output will be specific to the model you choose.  
4. Analyze and present your findings. An important part of creating predictive models is being able to communicate the results. In this final step of the project, you will present your findings using slides or an infographic. Your product should include the following components:
- Your reasoning for the algorithm you chose
- An explanation and analysis of the output of your model: What results did your model produce? What do they mean?
- A prediction based on your model
- A summary of the accuracy of your model
- Real world implications

## Choose a Dataset

For this project, you can work in groups of 4 or less.  Once you have determined your group, then decide what dataset you would like to work with.  You can find datasets on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/), [Kaggle](https://www.kaggle.com/), or you can find another source of your liking.  These are huge datasets, so make sure that you can work on the dataset and try to find something that is appropriate for your team to work with.  What could be the challenges?  Are there missing pieces of information?  Are there categories?  Is there data that you won't need?  Do you have to preprocess any of the data?  If so, how are you going to do it, and why?  These are all questions that you might want to consider.  Also, come up with a hypothesis on what you think your findings will tell you.  Have an idea of why you are using this data and what it will tell you about your situation.  You'll be presenting your findings to the class on Wednesday.

You'll need to create at least one file in this repository with your python project file, but you might need to creat another file with your data and may want to create a file with your graphs and such.  You can use a traditional file, or if you want to explore with Jupyter Notebooks, that is fine as well.

## Choosing An Algorithm

Now that you’ve chosen your dataset, it’s time to choose which algorithm you will use. Throughout this module, you have studied the following algorithms:

- Linear regression
- Logistic regression (classification)
- K-means (clustering)
One of the most challenging parts of mastering machine learning is choosing the right algorithm. This decision should be guided by the dataset as well as your goal for analyzing the data:

- Is there a linear correlation between variables?
- Do you want to predict a numerical value?
- Are you wanting help structuring the data into groups?
These are the types of questions you will need to ask yourself as you choose which algorithm is best for your chosen dataset.

The first step is to get familiar with your data. What variables are included in the dataset? What is the driving question or purpose of your model? Once you have an understanding of these two aspects of your dataset, you can start the process of choosing the right algorithm.

<b>Supervised vs. Unsupervised Learning Algorithm</b>

When choosing between an unsupervised and supervised algorithm, there is one key question to consider: Do you know the label, or outcome, for your output? For example, when you were working with the car prices dataset, you knew the outcome you were looking for was the price of the car. When you looked at the SUV dataset, you knew the outcome you were looking for a specific category: will buy an SUV and will not buy an SUV. If this is the case, you will want to use a supervised learning algorithm.

On the other hand, when you analyzed the mall customer data, you did not know the label for output. Instead, the computer created the groups for you and you created a label based on the computer-generated groups. If this is the case, then will you want to use an unsupervised learning algorithm.

<b>Linear vs. Logistic Regression</b>
If you decided on a supervised learning algorithm, the next step is to identify if you should use linear or logistic regression. Linear regression models are used for predicting numerical values, such as the price of a car or the time needed to travel between two places. Logistic regression models are used for classification, or when you want to label each output as part of a specific group.

## Present Your Project to the Class

On Wednesday January 8th, we will be presenting our project idea to the class.  You may have already started working on something, which is great, but I would like you to do a short one minute presentation to the class on what your dataset is and what algorithm you are going to use.  Please include your hypothesis as well.  If you have graphs already, then please include these in your presentation.  This should only take half of the class period.

## Program your Model and Analyze the Result

This will be the majority of your time in class on Thursday - Final.  Work on the model and come up with a solution.  Hopefully it is accurate, but if it is not, that is okay.  I am looking for you to create a model and find if your hypothesis is true or false.  You have all class periods to complete this portion.

## Present your Model

During the 2nd half of the final class period, you will be presenting your model to the class.  You can either create a slides presentation or just show your program and talk about it.  Here are the things that I would like to see in each presentation:
1. What was your dataset and where did you get it from (source).
2. Did you have to do any preprocessing to your dataset? If so, why did you choose what you did?
3. Graphs showing the relationship and why you chose whichever model you decided to use.
4. Show the program of your model.
5. What were the results of your testing data. 
6. What went well, what did you struggle on? 
7. What were your final conclusion of your project.

There will be a grade for your initial project presentation on Wednesday out of 5 points and then your final presentation will be out of 10 points.  You will also have a grade of 10 points for the actual program, which should be pushed to this github repository by the end of the period on the day of the final.
