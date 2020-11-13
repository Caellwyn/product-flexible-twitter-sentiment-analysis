# NLP_Tweet_Sentiment_Analysis

![image of tech products](reports/figures/ady-teenagerinro-tech-products-unsplash.jpg)

image courtesy of Ady Teenagerinro and unsplash

### What's the Buzz?
Many companies want to keep track of public sentiment about various products.  Positive sentiment might drive production for creators or inventory management for retailers.  It might drive the attention of influencers, news organizations, and consumers.  Twitter is a platform where individuals and companies express themselves and have conversations and is full of information about how people feel about particular products.

Our goal in this project is to create a model that can 'read' tweets to determine whether they express a positive, negative, or neutral sentiment about a target product.  We have generalized this model to work for any target an interested party may be interested in by training it to be product flexible.  It uses a user-supplied target product to determine the sentiment about that product in particular and determine the sentimental relationship between products within a tweet.  For instance, if the target product is 'Ipad', then the tweet, "Ipads are better than Android tablets" will be categorized as positive.  On the other hand, if the target is 'Android' then model would recognize that as a negative tweet tweet about Android products.

# The Data

We trained our model using 4 datasets hosted on data.world's [Crowdflower](https://data.world/crowdflower): 

* The Brands and Product Emotions dataset 

* The Apple Twitter Sentiment dataset 

* The Deflategate Sentiment dataset

* The Coachella-2015 dataset

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

Our main goal for our project is to create a model that will predict sentiment relative to a flexible product.  To do this, we do not want our model to make predictions based on user's sentiments based on prior tweets, but instead to judge the tweet solely on the tweet sentence content.  In other words, we do not want our model to naturally predict tweets about iPhones to be positive when we have a lot of existing tweets with users that sing praises about their new iPhones.  To do so, we will deal with it by replacing any instances of a tweet's target in their respective text with one, all-encompassing phrase "product_target".  Thus instead of saying "I love my new iPhone.", it will become "I love my new product_target.".  By doing this our model will be more focused on the "I love my" portion of the sentence istead of whether or not existing data supports positive tweets for an iPhone. Not only does this help deal with overfitting, this would allow our model to work on targets outside of the ones in our dataset, such as if we wanted to look at tweets about a festival or a sports team.

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



# The Models
We explored 3 very different model architectures: shallow models, convolutional neural networks, and recurrent neural networks.

## Shallow Models

# Elena Add More Here

## Convolutional Neural Networks

# Matt Add More Here

## Recurrent Neural Networks.

We began with a rather simple RNN using a trainable embedding layer followed by a long-short term memory layer, a max pooling layer to process the sequence data from the LSTM, a densely connected layer and finally a smaller dense output layer.  This model performed reasonably well, achieving a 62% accuracy on the validation data.

We also explored using pretrained GloVe Embeddings<sup>1</sup> rather than a trainable embedding layer, but because so much of the vocabulary from our dataset was nonstandard English, the GloVe dictionary, though massive woulc only encode about 2/3rds of it.  The model lost too much information from the tweets to be able to perform well.  We also tried initializing the embedding layer with the GloVe embeddings and then letting train from there, but this also gave disappointing results.

The change that did end up making some difference was to make the LSTM layer both smaller (from 25 neurons to 5 neurons) and to make it bidirectional so it read the tweets both forward and backward at the same time.  These changes raised the accuracy from 62% to 64.5% with a reasonable distribution of errors across classes.  It did somewhat better on positive emotion labeled tweets, and tended to miscategorize neutral tweets as negative.  This ended up being as much accuracy as we could squeeze out of this style of model.

# Summary

We created a model that can label tweets about a specific product with 67% accuracy, about 2/3rds of the time.  This can be used, in conjunction with a twitter crawler, to determine how positively or negatively a specific product is being regarded in the Twitterverse during a specific timeframe.  The best model we made was also the simplest, fastest, and least computationally demanding, a logistic regression model.

# Going Forward:

Future researchers might explore better ways to lemmatize and prepare twitterspeak to be more semantically accessible to predictive models. They also might try using other kinds of neural networks, such a gated recurrent unit models or simpler densely connected feed forward models.  While complex deep learning solutions may improve on our work, it's also possible that tweets can best be classified using 'bag of words' style modeling.  In this case the specifics of word encoding might be what makes the difference and using GloVe embeddings to encode words for a shallow model could be productive.

# References:

<sup>1</sup>Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf). [pdf] [bib]

