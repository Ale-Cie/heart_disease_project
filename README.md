# Heart Disease ML Project

A complete workflow through the Cleveland heart disease dataset. From data analysis, through model creation to experimenting on improving the model.

This is my first actual notebook that I wanted to share. We will take a look at the slightly altered Cleveland UCI heart disease dataset, available on Kaggle under this <a href="https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci/code?select=heart_cleveland_upload.csv">link</a>. In it's current state it is not yet finished, but this will change in the next few weeks (hopefully). In this notebook I am attempting to go through an entire workflow as depicted below (author is Daniel Bourke, you can read about these steps right <a href="https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/">here</a> on his website)

<img src="https://cdn-images-1.medium.com/max/2400/1*Gf0bWgr2wst9A1XR5gakLg.png" title="The complete ML steps" />

Since the notebook was considered "done" I put it up on Kaggle, and you can see it right <a href="https://www.kaggle.com/code/aleksanderciesielski/heart-disease-ml-workflow">here</a>.

<b>Thanks to I.J. for making me post it early!</b>

<i>-- 26.02.2023 --</i>

Tough week so not much of an update - I reworked the second round so that it functions as it should, and we actually are quite a bit happier with the exported model. I checked to see if the features in Round 2 were used to their fullest potential and I initiated the third round by creating a new dataframe. Oh and I also Kaggle-proofed the code so that it won't break each time I upload it there and forget to add an url to the dataset. See you next time!

<i>-- 20.02.2023 --</i>

Another week another update. Round 2 is finished (for now at least) - I went over fine tuning the `LogisticRegression()` model with both `RandomizedSearchCV()` and `GridSearchCV()`. Then I recreated the best models that these functions found and went through all the metrics as in Round 1. Finally, I attempted to manually tune the parameters that were different between two best models to see, if changing them in some way makes any difference in performance. I also decided to dump two models, even though they both underperform.

<i>-- 13.02.2023 --</i>

Some more changes added to almost all files. I changed the environment name in the enivronment.yml from a generic and bland "env" to "ML-env". The notebook has been updated with inspection of feature importances for both `LogisticRegression()` and `RandomForestClassifier()`. Round 1 of the modelling and tuning is done, now we're off to Round 2 and Round 3. Each of them will be dedicated to a single model that I'll attempt to tune to get as close to the accuracy goal as possible. I am also adding the notebook in the current state to Kaggle - you can visit it right <a href="https://www.kaggle.com/code/aleksanderciesielski/heart-disease-ml-workflow">here</a>.

<i>-- 07.02.2023 --</i>

I deleted the .gitignore file. We can almost say that the notebook is "done", although it still needs some changes, and more thorough tuning. I tuned the models with both RandomizedSearchCV and GridSearchCV, then I evaluated the models with both the classification report and cross-validated metrics. It's almost completed, after I'll finish the whole workflow I'll go back to the start and go into more small details.

<i>-- 05.02.2023 --</i>

The README.md was rewritten, a correct environment.yml was pushed.
The current steps that I (more or less) cover in this notebook are:

1. Problem definition
2. Data analysis
3. Evaluation
4. Features (kind of but not really)

Currently working on step 5. Modelling. So far I have created three models, LogisticRegression(), KNearestNeighbors() and RandomForestClassifier(). After experimenting with different values for n_neigbors I decided to abandon KNN as it seems to not be good enough of a model for this particular dataset.

<i>-- 04.02.2023 --</i>

The repository was created, unedited files were pushed.
