# NLP_Tweet_Sentiment_Analysis

![image of tech products](reports/figures/ady-teenagerinro-tech-products-unsplash.jpg)

image courtesy of Ady Teenagerinro and unsplash

### What's the Buzz?
Many companies want to keep track of public sentiment about various products.  Positive sentiment might drive production for creators or inventory management for retailers.  It might drive the attention of influencers, news organizations, and consumers.  Twitter is a platform where individuals and companies express themselves and have conversations and is full of information about how people feel about particular products.

Our goal in this project is to create a model that can 'read' tweets to determine whether they express a positive, negative, or neutral sentiment about a target product.  We have generalized this model to work for any target an interested party may be interested in by training it to be product flexible.  It uses a user-supplied target product to determine the sentiment about that product in particular and determine the sentimental relationship between products within a tweet.  For instance, if the target product is 'Ipad', then the tweet, "Ipads are better than Android tablets" will be categorized as positive.  On the other hand, if the target is 'Android' then model would recognize that as a negative tweet tweet about Android products.

# The Data

We trained our model using 3 datasets hosted on data.world's [Crowdflower](https://data.world/crowdflower): 

* The Brands and Product Emotions dataset 

* The Deflategate Sentiment dataset

* The Coachella-2015 dataset

# Andrew add more here

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

