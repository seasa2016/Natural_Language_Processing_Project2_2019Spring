# Bidirectional GRU + CNN based models

1. Pre-processing on each tweet
    - Replace "’" with "'" in each tweet
    - Expand Contractions
    - Convert each words into lowercase in each tweet.
    - Use regular experssion to remove unusual letters
    - Remove 'url' from each tweet
    - Use TweetTokenizer to represent each tweet with corresponding tokens
    - Perform spell checking on each processed tweet
    - Use WordNetLemmatizer to represent each tweet with corresponding lemmatizations (POS = "v")

2. Run different models in each task
    - BiGRU-CNN (w/ pre-trained embedding)
    - BiGRU-CNN (w/o pre-trained embedding)
    - CNN (w/o pre-trained embedding)
    - BiGRU + attention layer (w/ or w/o pre-trained embedding)

## NOTE
1. Enviroment: python3
    - wordcloud==1.5.0
    - Keras==2.2.4
    - nltk==3.4.3
    - numpy==1.16.4
    - pandas==0.24.2
    - scikit-learn==0.21.2
    - scipy==1.2.1
    - sklearn==0.0
    - tensorflow==1.13.1
    - imblearn==0.0
    
2. trained models: https://drive.google.com/drive/u/0/folders/1XUr3BLrcNUZExFSjDZy35qdWpSlE492s

