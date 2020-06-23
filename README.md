# Udacity
This repo contains work towards the [Udacity](https://www.udacity.com/) Data Science NanoDegree.  It is work in progress split by Project.

## [Project 1 - Intro to DS](Project_1__Board_Games) 
<a href="https://medium.com/@paulgstubley/bored-games-c31340859bef?source=friends_link&sk=ed9a21aa4b75262a4a46b7dec87a9df6" target="_blank"><img alt="Medium logo" src="images/medium.png" align="right" height="45px"></a> Covers Exploratory Data analysis of an interesting data set, a simple LinearRegression model fit, and creating content to upload to a Medium post.

## [Project 2 - Software Engineering](Project_2__Software_Engineering)
### Introduction to OOP
<a href="https://pypi.org/project/pgs-climbing/" target="_blank"><img alt="PyPi logo" src="images/pypi.png" align="right" height="45px"></a>
Covers the foundations of OOP including:
- creating a class, and two daughter-classes
- packaging the package correctly
- uploading to PyPi (I left mine at test-PyPi, rather than cluttering the PyPi index, but I uploaded my [Climbing](https://github.com/paul-stubley/Climbing) package to Pypi to test these new skills)

### Introduction to Web Development

<a href="https://pgs-ny-collisions.herokuapp.com/" target="_blank"><img alt="Heroku logo" src="images/heroku.jpg" align="right" height="45px"></a>
Covers the foundations of HTML, css, javascript, bootstrap, plotly (JS and Python) and deploying a Flask app using Heroku.

The output can be found [here](https://pgs-worldbank-app.herokuapp.com/)

## [Project 3 - Data Engineering](Project_3__Disaster_Response_Pipeline)

### Disaster Response ML pipeline & App

This project covers the ETL & ML pipelines for a disaster recovery app, as well as deploying as a local Flask app.

Taking labelled direct messages, and those from social-media & news, at the time of a disaster, can we classify unlabelled messages into categories (e.g. Fire, Infrastrucure, Medical Assistance etc.) to pass on to emergency & support services.

## [Project 4 - Experimental Design & Recommendations](Project_4__Recommendation_Engines)

This project covers using Rank-based, Content-Based (WIP) and Matrix-Factorisation/User-User Collaborative Filtering based methods to recommend articles for users of the IBM platform.

## [Project 5 - Capstone Project](Capstone_Project)

This project covers using the Yelp 2016 review set to recommend restaurants to users using various methods.  The aim is to recommend restaurants in a more personalised way than simply suggesting the top rating restaurants in a given city.  A blog writeup of the work here can be found on [Medium](https://medium.com/@paulgstubley/personalised-restaurant-recommendations-using-funksvd-3beff200b01c?source=friends_link&sk=08979761b0ece6de2c965b2880e048f8).

After some optimisation of the fitting parameters, we are able to predict ratings slightly more personalised than simply predicting the average-restaurant rating.

The issues of skewed and imbalanced training ratings are briefly discussed.

We also create a reasonably generic recommender python class which can be used for any ratings dataset that is 'one-row-per-review' to carry out recommendations based on top-rated and collaborative filtering using FunkSVD.  A minimal example of using this class is included in the `recommender.py` file's `main()` function.

## ToDo

- [ ] Upload Project 4 Parts 1 & 2
- [ ] Optimise 4.1 Starbucks task
- [x] Create class for Recommender (and upload to PyPi?)
- [ ] Add hero images to this readme
- [ ] Adjust NY-collisions app so that the data is saved to improve loadtime
- [ ] Add NYC user input options
