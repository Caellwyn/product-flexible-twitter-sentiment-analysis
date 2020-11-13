# The Data

We trained our model using 4 datasets hosted on data.world's [Crowdflower](https://data.world/crowdflower): 

* The Brands and Product Emotions dataset 

* The Apple Twitter Sentiment dataset 

* The Deflategate Sentiment dataset

* The Coachella-2015 dataset

# Andrew add more here

# Data Exploration

| Index                 | Tweet Text                     | Tweet Target                                      | Tweet Sentiment                                                         |
| ----------------------|--------------------------------|---------------------------------------------------|-------------------------------------------------------------------------|
| Each row is one tweet | Contains the text of the tweet | Contains the target the tweet is directed towards | Contains the sentiment of the tweet, can be positive, negative, or none |

Our base data, which is the Brands and Product Emotions dataset, comes with three columns with categorical values, with every row being a separate tweet.  The first column contains the text of the tweet.  This text is copied striahgt from the original source, so it contains all the little unique things that tweets such as '@' mentions and '#' hashtags.  The second column contains the target of the tweet.  The target can be subjects such as 'iPad', 'iPhone', or 'Google' to name a few, and they denote what their respective tweet's subject is.  For example, if the tweet target is 'Android App', we can assume that the tweet text in the same row has something to do with it.  The last column contains the tweet sentiment, or the type of emotion the tweet text is showing.  There are three possible values: positive, negative, and no emotion.  A positive value would symbolize that the tweet has a positive feeling towards their listed target, while the opposite would be true if it was negative.  A value of no emotion would mean that the tweet does not have a particularily strong feeling towards either side.  As the tweet sentiment is the value that we are trying to predict, this will be our target column in our predictive models.

# Class Imbalance

To ensure that our model is accurate, we'll have to check the class disparity of our targets.  As we are setting the tweet sentiment as our target, we must check how many of each different target type that we have.

![Class Imbalance Before](/reports/figures/tweets_per_emotion_before.png)

As seen above, the class disparity is quite severe.  Ignoring the fact that the 'no emotion' values are almost double the amount of 'positives', the number of 'negative' sentiment tweets is far too low in order to properly class the tweets.  This is bad as it could lead to our model learning to get a high accuracy score simply by guessing 'no emotion' for any input.  As we do not want our model to overfit, we must deal with it.  While oversampling our 'negatives' or undersampling our 'no emotion' tweets is possible, we decided to instead import more data from similar datasets, which are datasets that must 1. contain rows of tweet texts and 2. contain a variable relating to the tweet sentiment.  We found three extra datasets that fit our requirements, the Apple Twitter Sentiment dataset, the Deflategate Sentiment dataset, and the Cochella-2015 dataset.  After we imported the extra data, processed them to match our base dataset, and concatenated them together, our class disparity becomes the following:

![Class Imbalance After](/reports/figures/tweets_per_emotion_after.png)

With the introduction of our new data, the difference in the number of each class becomes much smaller.  Both 'positive' and 'negative' tweets now have a similar amount, while 'no emotion' tweets follow closely behind.  Thus we do not have a need to undergo any over or undersampling.

# Feature Engineering

Our main goal for our project is to create a model that will predict sentiment relative to a flexible product.  To do this, we do not want our model to make predictions 
Product_target

![product_target_before_after](/reports/figures/product_target_before_after.png)


# Data Cleaning and Preprocessing

As our tweet text comes straight from the source, they are undoubtedly very messy.  We must first preprocess the data in order to convert them to a form safe for consumption by our predictive models.  The following is a list of processes we took to turn our dirty source tweets into clean, filtered data:

<ol>
<li>Split the tweet into tokens</li>
<li>Convert all capitalized letters into lower-case</li>
<li>Remove punctuation</li>
<li>Remove twitter jargon such as @ mentions</li>
<li>Remove leftover numbers</li>
<li>Remove words with accents</li>
<li>Remove stop words</li>
<li>Replace instances of the target in the text with 'product_target'</li>
<li>Remove empty strings</li>
<li>Lemmatize the words</li>
<li>Rejoin all the tokens into one string</li>
</ol>

Here is an example of a tweet looks like after we undergo cleaning on it:

![tweet_cleaning_before_after](/reports/figures/tweet_cleaning_before_after.png)

After going through every row and applying our cleaning function to it, we will drop our target column as we have no more need of it.  Thus our resulting dataset after pre-processing will look like the follwing:

| Index                 | Tweet Text                             | Tweet Sentiment                                        |
| ----------------------|----------------------------------------|--------------------------------------------------------|
| Each row is one tweet | Contains the cleaned text of the tweet | Contains the sentiment of the tweet, can be 0, 1, or 2 |

