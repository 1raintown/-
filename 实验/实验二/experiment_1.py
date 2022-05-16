#!/usr/bin/env python
# coding: utf-8

# 
# <h1 align=center><font size = 5> <center>文本分析与挖掘</center> </font></h1> 
# 
# <h2 align=center><font size = 4><center>实验一、文本预处理和基本表示-Part 1</center></font></h2>
# <h2 align=center><font size = 2><center>浙江工业大学计算机科学与技术学院 杨宇圳</center></font></h2>

# ## 实验内容
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# <li>英文常用预处理方法</li>
# <li>中文分词</li>
# <li>分句</li>
# 

# ### <b> 1. 英文预处理</b>

# <b>a.导入包</b>

# In[1]:


import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

ps=PorterStemmer()#词干提取
wnl = WordNetLemmatizer() #词性还原
stopwords_list=nltk.corpus.stopwords.words('english')


# <b>b.编写英文预处理函数 EngPreprocess(),对输入的一个英文段落实现以下功能：分词、词干提取、词性还原、去停用词。</b>

# In[2]:


def get_wordnet_pos(tag):#获取词性
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def remove_stopwords (text):#去停用词
    filtered_word=[i for i in text if i.lower() not in stopwords_list ]
    #filtered_text=' '.join(filtered_word)
    return filtered_word    

#编写函数EngPreprocess
def EngPreprocess(input_text, tokenization=False,
                  stemming=False, lemmatization=False,stopword_remove=False):
    if tokenization :#分词
        input_text=word_tokenize(input_text)
    if stemming :#词干提取
        #input_text=word_tokenize(input_text)
        input_text=[ps.stem(i) for i in input_text]
        #input_text=' '.join(input_text)
    if lemmatization:#词性还原
        lemmas_sent = []
        #input_text=word_tokenize(input_text)
        tagged_sent = pos_tag(input_text)     # 获取单词词性
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))# 词形还原
            
        #input_text=' '.join(lemmas_sent)
    if stopword_remove:#去停用词
        #input_text=word_tokenize(input_text)
        input_text=remove_stopwords(input_text)
        
    return input_text

a="The quick brown fox jumps over the lazy dog."
result1=EngPreprocess(a,True,stopword_remove=True)
print(result1)
# <b>c.对比text.split()与分词结果；</b>

# 用以下几个句子作为测试：
# <li>句子1：UK coronavirus cases top 600,000 as another 12,872 confirmed</li>
# <li>句子2：Global COVID-19 cases surpass 38 mln: Johns Hopkins University</li>

# In[3]:


#对以上例句分别调用EngPreprocess()中的分词功能，text.split()得到结果

input_text="UK coronavirus cases top 600,000 as another 12,872 confirmed"
print("样例一：",input_text)
test_text1=EngPreprocess(input_text, tokenization=True)
print('EngPreprocess()结果1:',test_text1)
print('text.split()结果1:',input_text.split())
print()
input_text="Global COVID-19 cases surpass 38 mln: Johns Hopkins University"
print("样例二：",input_text)
test_text1=EngPreprocess(input_text, tokenization=True)
print('EngPreprocess()结果2:',test_text1)
print('text.split()结果2:',input_text.split())


# <div class="alert alert-success alertsuccess" style="margin-top: 10px">
# 结果分析和讨论：自定义的分词函数与text.split()的结果大致相似，而text.split()只会根据空格来分词，“mln:”这里就未分开，而自定义的分词函数则能很好地完成任务。

# <b>d.对词干提取、词性还原、去停用词分别进行测试（用多个不同样例），观察结果并讨论准确性。</b>

# 用以下样例测试词干提取功能
# <li>jumping</li>
# <li>jumps</li>

# In[4]:


input_text="jumping"
print("样例一：",input_text)
print(EngPreprocess(input_text, stemming=True))

input_text="jumps"
print("样例二：",input_text)
print(EngPreprocess(input_text, stemming=True))


# <div class="alert alert-success alertsuccess" style="margin-top: 10px">
# 结果分析和讨论：查看结果，发现可以实现对词干提取功能

# 用以下样例测试词性还原功能
# <li>There are a lot of cars</li>
# <li>Whatever is worth doing is worth doing well.</li>

# In[5]:


input_text="There are a lot of cars"
print("样例一：",input_text)
print(EngPreprocess(input_text, lemmatization=True))
input_text="Whatever is worth doing is worth doing well."
print("样例二：",input_text)
print(EngPreprocess(input_text, lemmatization=True))


# <div class="alert alert-success alertsuccess" style="margin-top: 10px">
# 结果分析和讨论：对照结果，发现可以实现词性还原的功能，且准确性较高

# 用以下样例测试去停用词功能
# <li>The,and,if are stopwords,computer is not</li>
# <li>Love is not a maybe thing. You know when you love someone.</li>

# In[6]:


input_text="The,and,if are stopwords,computer is not"
print("样例一：",input_text)
print(EngPreprocess(input_text,stopword_remove=True))
input_text="Love is not a maybe thing. You know when you love someone."
print("样例二：",input_text)
print(EngPreprocess(input_text,stopword_remove=True))


# <div class="alert alert-success alertsuccess" style="margin-top: 10px">
# 结果分析和讨论：对照结果，发现可以实现去停用词的功能，且准确性较高

# ### <b> 2. 中文分词</b>

# #### a.导入包

# In[7]:


import jieba


# #### b.编写函数 ChTokenize(),基于 jieba 实现中文分词。
# 

# In[8]:


def ChTokenize(input_text,cut_all=False):
    seg_list=jieba.cut(input_text,cut_all)
    return seg_list


# 用以下样例测试中文分词功能
# <li>人生就是一场旅行，不在乎目的地，在乎的应该是沿途的风景以及看风景的心情。</li>
# <li>用以下样例测试中文分词功能。</li>

# #### c.测试

# In[9]:


input_text="人生就是一场旅行，不在乎目的地，在乎的应该是沿途的风景以及看风景的心情。"
print("样例一：",input_text)
print(u"精确模式：",'/'.join(ChTokenize(input_text)))
print(u"全模式：",'/'.join(ChTokenize(input_text,True)))
print()

input_text="用以下样例测试中文分词功能。"
print("样例二：",input_text)
print(u"精确模式：",'/'.join(ChTokenize(input_text)))
print(u"全模式：",'/'.join(ChTokenize(input_text,True)))


# <div class="alert alert-success alertsuccess" style="margin-top: 10px">
# 结果分析和讨论：精确模式下，分词的结果较符合语义的分词，而全模式下则将全部可能出现的词都进行了分隔

# ### 3.分句

# #### a.编写函数 Doc2Sent(),实现对英文、中文文档进行分句。

# In[10]:


import re
from nltk.tokenize import sent_tokenize

def merge_symmetry(sentences, symmetry=('“','”')):
    '''中文拼接对称符号'''
    effective_ = []
    merged = True
    for index in range(len(sentences)):       
        if symmetry[0] in sentences[index] and symmetry[1] not in sentences[index]:
            merged = False
            effective_.append(sentences[index])
        elif symmetry[1] in sentences[index] and not merged:
            merged = True
            effective_[-1] += sentences[index]
        elif symmetry[0] not in sentences[index] and symmetry[1] not in sentences[index] and not merged :
            effective_[-1] += sentences[index]
        else:
            effective_.append(sentences[index])
        
    return [i.strip() for i in effective_ if len(i.strip()) > 0]

def Doc2Sent(paragraph,mode):
    if mode=='Chinese':
        """中文分句"""
        sentences = re.split(r"(？|。|！|\…\…)", paragraph)
        
        sentences.append("")
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        sentences = [i.strip() for i in sentences if len(i.strip()) > 0]
        #进行 ” 的拼接
        for j in range(1, len(sentences)):
            if sentences[j][0] == '”':
                sentences[j-1] = sentences[j-1] + '”'
                sentences[j] = sentences[j][1:]

        return merge_symmetry(sentences)
    elif mode=="English":
        '''英文分句'''
        sent_tokenize_list = sent_tokenize(paragraph)
        
        return sent_tokenize_list
    else :
        print("模式选择有误，重新输入")
        return "请重新输入"



# #### b.测试

# 中文样段：
# 
# &emsp;&emsp;这问题很简单。因为如果泛指“人类”，那就是站在生物的立场。结婚，是为了传宗接代。虽然说，不结婚而同居或只发生性的关系，照样可以传宗接代。但那样实在对所要传的后代很不方便。因为数千年来，人类根据经验，已经得到证明，要安全地抚养后代，实在不是单单的男方或女方独自的力量可做得好的，而必须双方分工合作，有人在家照顾哺育，有人出外打工谋生，才不会顾此失彼，疲于奔命。所以双方要建立一个固定的居所，有个固定的名分，负起人伦、社会与法律上的责任，使这关系巩固而公开，以免中途发生动摇或受到外力的侵害。这样才可以有效地保护及教养子女，使他们成为人类所希求的、良好的后代。所以，“人”需要结婚!
#     
#     
# 英文样段：
# 
# I remember quite clearly now when the story happened. The autumn leaves were floating in measure down to the ground, recovering the lake, where we used to swim like children, under the sun was there to shine. That time we used to be happy. Well, I thought we were. But the truth was that you had been longing to leave me, not daring to tell me. On that precious night, watching the lake, vaguely conscious, you said: Our story is ending.
# 

# In[11]:


C_paragraph="这问题很简单。因为如果泛指，我说“人类。就是如此。”那就是站在生物的立场。结婚，是为了传宗接代。虽然说，不结婚而同居或只发生性的关系，照样可以传宗接代。但那样实在对所要传的后代很不方便。因为数千年来，人类根据经验，已经得到证明，要安全地抚养后代，实在不是单单的男方或女方独自的力量可做得好的，而必须双方分工合作，有人在家照顾哺育，有人出外打工谋生，才不会顾此失彼，疲于奔命。所以双方要建立一个固定的居所，有个固定的名分，负起人伦、社会与法律上的责任，使这关系巩固而公开，以免中途发生动摇或受到外力的侵害。这样才可以有效地保护及教养子女，使他们成为人类所希求的、良好的后代。所以，“人”需要结婚!"
print('中文分句：')
print('\n'.join(Doc2Sent(C_paragraph,'Chinese')))
print()
E_paragraph="I remember quite clearly now when the story happened...... The autumn leaves were floating in measure down to the ground, recovering the lake, where we used to swim like children, under the sun was there to shine. That time we used to be happy. Well, I thought we were. But the truth was that you had been longing to leave me, not daring to tell me. On that precious night, watching the lake, vaguely conscious, you said: Our story is ending."
print('英文分句：')
print('\n'.join(Doc2Sent(E_paragraph,'English')))


# <div class="alert alert-success alertsuccess" style="margin-top: 10px">
# 结果分析和讨论：中文段落和英文段落都准确地进行了分句
