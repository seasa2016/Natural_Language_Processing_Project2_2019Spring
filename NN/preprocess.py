import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import sys
import operator
from tqdm import tqdm

data = pd.read_csv(sys.argv[1], sep='\t')

def remove_emoji(sen):
    sen = str(sen).lower()
    temp = ['']
    for word in sen.strip().split():
        if(temp[-1] == '@user' and word == '@user'):
            continue
        if(word[0]=='#'):
            continue
        temp.append(word)

    sen = ' '.join(temp[1:])
    """
    emoji_pattern = re.compile("["	
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

    sen = emoji_pattern.sub(r'', sen)
    sen = re.sub(r'[\u3000-\u303F]', '', sen)
    sen = re.sub(r'[\u3000-\u303F]', '', sen)
    sen = re.sub(r'[\uFF00-\uFFEF]', '', sen)

    emoji = '💕' + '🤘' + '👯' + '🏋️' + '🚗🚗' + '💪' + '🎄' + '🔁' + '👋' + '💪💪' + '🔥' + \
    '😝👶🏻☀️🍹' + '🔁' + '✨👯' + '🌞' + '💖' + '🎉✨' + '🔁' + '🌸' + '❤️💛💚💙💜' + \
    '🇫🇷' + '🎆' + '💋' + '🔁' + '👂' + '🎸' + '💦💦' + '💦💦' + '❤' + '🍷' + \
    '🎂' + '👫' + '🌞' + '🌊' + '🍁' + '🔁' + '🎶' + '🌹🌹' + '😆' +'🐔' + '🤠🇹🇭' + \
    '👯' + '🌙' + '🎄' + '🎤' + '🔥🔥' + '🔥🔥' + '📚' + '💕' + '🙋' + '😁👾🍸' + '🔥' + \
    '🎓' + '🍷' + '✏️🍍🍎✏️' + '💧' + '🔁' + '🌞' + '🛋' + '🔁' + '💪🏼' + '💔' + \
    '🔁' + '🏆' + '🏆' + '🔁' + '🎉' + '🎉' + '🏃👟' + '🍖🍷🍾' + '❤️' + '🔁' + '😢' + \
    '👋🏼' + '👂' + '🗽' + '🇵🇷' + '🎸' + '🔁' + '🌧' + '🔁' + '🍿🍿' + '❤️' + '😈' + \
    '😈' + '🇭🇰' + '🌞' + '🌞' + '😜' + '👊' + '🇺🇸' + '😂' + '🤔' + ''

    for c in emoji:
        sen = sen.replace(c, '')
    """2
    noises = ['url', '\'ve', 'n\'t', '\'s', '\'m']
    sen = sen.replace('url', '')
    sen = sen.replace('\'ve', ' have')
    sen = sen.replace('\'m', ' am')
    sen = sen.replace('@user', 'user')

    return sen

mispell_dict = {'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not' ,
                'wasnt': 'was not' ,
                'hasnt': 'has not' ,
                '‘i': 'i' ,
                'theatre': 'theater' ,
                'cancelled': 'canceled' ,
                'organisation': 'organization' ,
                'labour': 'labor' ,
                'favourite': 'favorite' ,
                'travelling': 'traveling' ,
                'washingtons': 'washington' ,
                'marylands': 'maryland' ,
                'chinas': 'china' ,
                'russias': 'russia' ,
                '‘the': 'the' ,
                'irans': 'iran' ,
                'dulles': 'dulle',
                'labour': 'labor',
                'metoo':'me too'
                }

def parse(text):
    #build up inverted file
    for punct in "/-'":
        text = text.replace(punct, '')
    for punct in '?!.,"#$%\'()*+-/:;<=>[\\]^_`{|}~' + '“”’‘':
        text = text.replace(punct, ' ')
    """
    if you want to parse number
    text = str(key)
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    """
    for key,data in mispell_dict.items():
        text = text.replace(key,data)
    return text

data['tweet'] = data['tweet'].apply(lambda x : remove_emoji(parse(x.lower())))
