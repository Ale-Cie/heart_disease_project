# Heart Disease ML Project

A complete workflow through the Cleveland heart disease dataset. From data analysis, through model creation to experimenting on improving the model.

This is my first actual notebook that I wanted to share. We will take a look at the slightly altered Cleveland UCI heart disease dataset, available on Kaggle under this <a href="https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci/code?select=heart_cleveland_upload.csv">link</a>. In it's current state it is not yet finished, but this will change in the next few weeks (hopefully). In this notebook I am attempting to go through an entire workflow as depicted below (author is Daniel Bourke, you can read about these steps right <a href="https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/">here</a> on his website)

<img src="https://cdn-images-1.medium.com/max/2400/1*Gf0bWgr2wst9A1XR5gakLg.png", title="The complete ML steps" />

When the notebook will be finished I will post it on Kaggle as well. Until then let's get to work.

<b>Thanks to I.J. for making me post it early!</b>

-- 05.02.2023 --
The README.md was rewritten, a correct environment.yml was pushed.
The current steps that I (more or less) cover in this notebook are:

1. Problem definition
2. Data analysis
3. Evaluation
4. Features

Currently working on step 5. Modelling. So far I have created three models, LogisticRegression(), KNearestNeighbors() and RandomForestClassifier(). After experimenting with different values for n_neigbors I decided to abandon KNN as it seems to not be good enough of a model for this particular dataset.

-- 04.02.2023 --
The repository was created, unedited files were pushed.
