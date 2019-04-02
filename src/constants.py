"""
 * @Author: radsn23
 * @Date: 2019-03-07 12:49:08
 * @Last Modified by:   radsn23
 * @Last Modified time: 2019-03-07 12:49:08
"""

"""
Contains the paths for all the files used and generated
"""

HOME = '../'
TRAIN_POS = '/WSJ_02-21.pos'
VALIDATE_POS = '/WSJ_24.pos'
VALIDATE_WORDS = '/WSJ_24.words'
TEST_WORDS = '/WSJ_23.words'
TEST_POS = '/WSJ_23.pos'

VOCAB = HOME + 'data/vocab.txt'
STATS = HOME + 'data/stats.txt'

PRED_V_POS = HOME + 'WSJ_24.pos'
PRED_T_POS = HOME + 'WSJ_23.pos'
#Assign tags for open classes using suffixes
"""
https://www.thoughtco.com/common-suffixes-in-english-1692725
https://dictionary.cambridge.org/us/grammar/british-grammar/word-formation/suffixes
https://wac.colostate.edu/books/sound/chapter5.pdf
"""
NOUN_SUFFIX = ['age','al','ance','ence','dom','ee','er','ation',\
        'or','hood','ism','ist','ty','ment','ness','ry'\
        'ship','ion','cy','ling','scape','let','ette','iana','ess']
VERB_SUFFIX = ['ate','en','fy','ize','ise']
ADJ_SUFFIX = ['able','ible','al','esque','ful','ic','ical','ous',\
        'ish','ive','less','y','ian','less','ese','atory']
ADV_SUFFIX = ['ly','ward','wise','wards']

RARE_TAGS = ['PUNCT','DIGIT','CAPS','RARE_NOUN','RARE_VERB',\
        'RARE_ADJ','RARE_ADV','RARE_WORD']

#ALPHA = 1
#ALPHA = 0.5
#ALPHA = 0.25
#ALPHA = 0.1
#ALPHA = 0.01
#ALPHA = 0.001
ALPHA = 0.005
