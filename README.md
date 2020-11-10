# NLP_Tweet_Sentiment_Analysis

### What's the Buzz?
Many companies want to keep track of public sentiment about various products.  Positive sentiment might drive production for creators or inventory management for retailers.  It might drive the attention of influencers, news organizations, and consumers.  Twitter is a platform where individuals and companies express themselves and have conversations and is full of information about how people feel about particular products.

Our goal in this project is to create a model that can 'read' tweets to determine whether they express a positive, negative, or neutral sentiment about a target product.  We have generalized this model to work for any target an interested party may be interested in by training it to be product flexible.  It uses a user-supplied target product to determine the sentiment about that product in particular and determine the sentimental relationship between products within a tweet.  For instance, if the target product is 'Ipad', then the tweet, "Ipads are better than Android tablets" will be categorized as positive.  On the other hand, if the target is 'Android' then model would recognize that as a negative tweet tweet about Android products.

### Data

We trained our model using the Brands and Product Emotions dataset hosted on data.world's [Crowdflower](https://data.world/crowdflower/brands-and-product-emotions) site.  

This dataset contain 3 variables describing 9093 tweets: the target product, the text of the tweet, and the emotional sentiment (human labeled).

2978 tweets express a positive emotion, 570 tweets express a negative emotion, 5389 tweets express no emotion, and human labelers were unable to determine the sentiments of 156 tweets.  We removed the unlabeled tweets, since there were a relatively small number and they would not be useful to our model's learning process.

The target products included in this dataset were:

iPad: 946 tweets, 125 negative, 24 no emotion, 793 positive
Apple: 661 tweets, 95 negative, 21 no emotion, 543 positive
iPad or iPhone App: 470 tweets, 2 negative, 1 no meotion, 32 positive
Google: 430 tweets, 68 negative, 15 no emotion, 346 positive
Iphone: 297 tweets, 203 negative, 9 no emotion, 184 positive
Other Google product or service: 293 tweets, 47 negative, 9 no emotion, 236 positive
Android App: 81 tweets, 8 negative, 1 no emotion, 72 positive
Android: 78 tweets, 8 negative, 1 no emotion, 69 positive
and Other Apple product or service: 35 tweets, 2 negative, 1 no emotion, 32 positive
No target product: 5802 tweets, 51 negative, 5297 no emotion, 306 positive


Phase 4 project using NLP techniques to perform sentiment analysis on a set of tweets

#placeholder for user oriented REAMME file
