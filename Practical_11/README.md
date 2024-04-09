# Introduction

This week's practical will focus on developing an NLP pipeline for classifying texts. The first task will follow an sklearn example using a toy dataset. 
The second task will allow you to create your own corpus and see if your ML pipeline can handle real world text. As always, there are bugs in both tasks.



## Task 1
If you want to see the original example, then follow the link: https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset 

```Python
from sklearn important metrics
form sklearn.datasets import fetch_20newsgroups

print(list(newsgroups_train.target_names))

from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

vectors_test = vectorizer.transform(newsgroups_test.data)

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestClassifier()

pred = clf.fit(vectors, newsgroups_train.target).predict_proba(vectors_test)
metrics.f1_score(newsgroups_test.target, pred)
```

## Task 2
Task 2 will allow you to retrieve abstracts from Pubmed without having to manually access the website. You will use the Biopython library, which
interfaces with Pubmed. The code will have you build a function that will extract 10 abstracts given a keyword. The idea is to build a structured
dataset of articles of different topics and then build a classifier that will automatically classify the articles by their topic. You may have
noticed that sciencedirect or web of science filters are inaccurate. If you have, then why not build your own, personalised filter?

```Python

# Allows you  to interface with Pubmed
!pip install biopython

import pandas as np
from Bio import Entrez

#Enter your email below
Entrez.email = "@qmul.ac.uk"

#Create a function for retrieving abstracts from Pubmed by using a keyword
definition fetch_abstracts(keyword, max_results=10):
    # Search for articles in PubMed
    search_handle = Entrez.esearch(db="pubmed", term=keyword, retmax=max_results)
    search_results = Entrez.read(search_handle)
    search_handle.close()

    # Fetch details of articles
    id_list = search_results["IdList"]
    fetch_handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
    fetch_results = Entrez.read(fetch_handle)
    fetch_handle.close()

    # Extract abstracts
    abstracts = []
    for article in fetch_results['PubmedArticle']:
        try:
            abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            abstracts.append(abstract)
        except KeyError:  # Abstract is missing
            abstracts.append("Abstract not available")

    print abstracts

# enter your keyword
keyword = ""

# Fetch abstracts
abstracts = fetch_abstracts(keyword)


# Change to DataFrame
df = pd.DataFrame(abstracts, columns=['abstract'])

# Assign a label to the instances
df['target'] = keyword

#Check the dataframe
df

#Now build your classifier

```
