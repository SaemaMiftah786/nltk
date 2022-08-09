from urllib.request import Request
from urllib.request import urlopen
from bs4 import BeautifulSoup

import re
import codecs
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, sent_tokenize

import pandas as pd
import readability

import spacy
from textstat.textstat import textstatistics,legacy_round

import os


 
#Useful Functions
def break_sentences(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return list(doc.sents)
 
#Number of Words in the text
def word_counting(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words
 
#The number of sentences in the text
def sentence_count(text):
    sentences = break_sentences(text)
    return len(sentences)
 
#Average sentence length
def avg_sentence_length(text):
    words = word_counting(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length
 

def syllables_count(word):
    return textstatistics().syllable_count(word)
 
#The average number of syllables per word in the text
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_counting(text)
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)
 
#Total Difficult Words in a text
def difficult_words(text):
     
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    #find all words in the text
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]
 
    #difficult words are those with more than 2 syllables
    
    diff_words_set = set()
     
    for word in words:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2:
            diff_words_set.add(word)
 
    return len(diff_words_set)

 
def gunning_fog(text):
    per_diff_words = (difficult_words(text) / word_counting(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
    return grade



def syllable_count(text):
    vowels = "aeiou"
    exceptions = ['es', 'ed']
    count = 0
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        wo = strip_suffixes(w, exceptions)
        for index in range(0, len(wo)):
            if wo[index] in vowels:
                count += 1
    return count

def strip_suffixes(s,suffixes): 
    for suf in suffixes: 
        if s.endswith(suf): 
            return s.rstrip(suf) 
    return s 



###############################
#Main Code
###############################

final_d = []
master_data = pd.read_excel('Input.xlsx')

for i, j in master_data.iterrows():
    url = j.URL
    url_id = j.URL_ID
    #print(url_id)
    
    #get the content
    raw_request = Request(url)
    raw_request.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0')
    raw_request.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8')
    resp = urlopen(raw_request)
    raw_html = resp.read()
    soup = BeautifulSoup(raw_html, 'html.parser')
    
    #get the title
    title_box = soup.find(class_='td-parallax-header')
    title = title_box.find("h1").get_text()
        
    #get the text
    text_box = soup.find(class_='td-post-content')
    text_list = text_box.find_all(["p", "h3"])
    text = ''
    t = ''

    for txt in text_list:
        t = re.sub(r' ?(\d+) ?', r' \1 ', txt.get_text())
        text += t.replace(":", " : ").replace(".", ". ").replace(")", " ) ").replace("/", " / ").replace('—', ' ').replace('-', ' ')

    
    
    #save the contents in a file
    file = codecs.open("./Text_Files/"+ str(int(url_id)) + ".txt", "w", "utf-8")
    file.write(title)
    file.write('\n')
    file.write(text)
    file.close()
    
    
    positive_dictionary = {}
    negative_dictionary = {}
    
    #sentimental analysis
    f = open("StopWords_Generic.txt", "r")
    
    df = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2021.csv')
    sentence = text
    
    word_token = word_tokenize(sentence)
    positive_score = 0
    negative_score = 0
    for word in word_tokenize(sentence):
        if word.upper() in f.read():
            word_token.remove(word)
        else:
            if word.upper() in df['Word'].values:            
                if(df[df['Word'] == word.upper()]['Positive'].values) > 0:
                    positive_dictionary[word.upper()] = (df[df['Word'] == word.upper()]['Positive'].values[0])
                    positive_score += 1

                elif(df[df['Word'] == word.upper()]['Negative'].values) > 0:
                    negative_dictionary[word.upper()] = (df[df['Word'] == word.upper()]['Negative'].values[0])
                    negative_score += -1


    negative_score = negative_score * (-1) 
    
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    
    total_words_after_cleaning = len(word_token)
    subjectivity_score = (positive_score + negative_score) / ((total_words_after_cleaning) + 0.000001)
    
    #analysis of readability
    gunning_fog_index = gunning_fog(text)
    
    #average number of words per sentence
    avg_number_of_words_per_sent = word_counting(text) / sentence_count(text)
    #complex word count
    complex_words = difficult_words(text)
    percentage_of_complex_words = (complex_words / len(word_token)) * 100
    
    
    #word count
    stop_words = set(stopwords.words('english'))
    puctuations = [ "?", "!", ",", "."]
    word_tokens = word_tokenize(text)
    
    filtered_sentence = []

    for w in word_tokens:
        if w.lower() not in stop_words:
            if w not in puctuations:
                filtered_sentence.append(w)
    

    #syllable count per word
    word_counts = len(filtered_sentence)
    syllable_count_per_word = syllable_count(text) / word_counts
    
    
    #personal pronouns
    pronouns = ["I", "we", "my", "ours", "us", "We", "My", "Ours", "Us"]
    pronoun_list = re.findall(r"(?=("+'|'.join(pronouns)+r"))", text)
    pronoun_list_len = len(pronoun_list)
    
    #average word length
    total_characters = len(text)
    average_word_length = total_characters / word_counts
    
    
    print(url_id)
    
    final_d.append(
        {
            'URL_ID': url_id,
            'URL': url,
            'POSITIVE SCORE': positive_score,
            'NEGATIVE SCORE': negative_score,
            'POLARITY SCORE': polarity_score,
            'SUBJECTIVITY SCORE': subjectivity_score,
            'AVG SENTENCE LENGTH': '',
            'PERCENTAGE OF COMPLEX WORDS': percentage_of_complex_words,
            'FOG INDEX': gunning_fog_index,
            'AVG NUMBER OF WORDS PER SENTENCE': avg_number_of_words_per_sent,
            'COMPLEX WORD COUNT': complex_words,
            'WORD COUNT': word_counts,
            'SYLLABLE PER WORD': syllable_count_per_word,
            'PERSONAL PRONOUNS': pronoun_list_len,
            'AVG WORD LENGTH': average_word_length
        }
    )
    
    

  

final_d = pd.DataFrame(final_d)
writer = pd.ExcelWriter('output.xlsx')
# write dataframe to excel
final_d.to_excel(writer, index=False)
# save the excel
writer.save()
    
    
    
