#!/usr/bin/env python3
# -*- coding: utf-8 -*-


################################## Importation des librairies ##################################

from gensim.summarization.summarizer import summarize
from collections import Counter
import pandas as pd
import datetime as dt
import string
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import bigrams
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from nltk.collocations import BigramCollocationFinder
from itertools import count
import praw
import urllib.request
import xmltodict   

################################## Déclaration des classes ##################################

class Corpus():
    
    def __init__(self,name):
        self.name = name
        self.collection = {}
        self.authors = {}
        self.id2doc = {}
        self.id2aut = {}
        self.ndoc = 0
        self.naut = 0
        self.result = Network(height="750px", width="100%", bgcolor="white", font_color="#333333")
        self.result_df = None
        self.N=15
        
        
        ################################## Création du Corpus ##################################
  
        reddit = praw.Reddit(client_id='QOyCM0Y3Uz3mcQ', client_secret='fGv7iEOm2z-9hPaSW7tGiKObrF67zQ', user_agent='Reddit WebScraping')
        hot_posts = reddit.subreddit('Coronavirus').hot(limit=100)
        for post in hot_posts:
                datet = dt.datetime.fromtimestamp(post.created)
                txt = post.title + ". "+ post.selftext
                txt = txt.replace('\n', ' ')
                txt = txt.replace('\r', ' ')
                doc = Document(datet,
                               post.title,
                               post.author_fullname,
                               txt,
                               post.url)
                self.add_doc(doc)
                
        url = 'http://export.arxiv.org/api/query?search_query=all:covid&start=0&max_results=100'
        data =  urllib.request.urlopen(url).read().decode()
        docs = xmltodict.parse(data)['feed']['entry']
                
        for i in docs:
                    datet = dt.datetime.strptime(i['published'], '%Y-%m-%dT%H:%M:%SZ')
                    try:
                        author = [aut['name'] for aut in i['author']][0]
                    except:
                        author = i['author']['name']
                    txt = i['title']+ ". " + i['summary']
                    txt = txt.replace('\n', ' ')
                    txt = txt.replace('\r', ' ')
                    doc = Document(datet,
                                   i['title'],
                                   author,
                                   txt,
                                   i['id']
                                   )
                    self.add_doc(doc)        
        
      
    def add_doc(self, doc):
        
        self.collection[self.ndoc] = doc
        self.id2doc[self.ndoc] = doc.get_title()
        self.ndoc += 1
        aut_name = doc.get_author()
        aut = self.get_aut2id(aut_name)
        if aut is not None:
            self.authors[aut].add(doc)
        else:
            self.add_aut(aut_name,doc)
            
    def add_aut(self, aut_name,doc):
        
        aut_temp = Author(aut_name)
        aut_temp.add(doc)
        
        self.authors[self.naut] = aut_temp
        self.id2aut[self.naut] = aut_name
        
        self.naut += 1  

    def get_aut2id(self, author_name):
        aut2id = {v: k for k, v in self.id2aut.items()}
        heidi = aut2id.get(author_name)
        return heidi

    def get_doc(self, i):
        return self.collection[i]
    
    def get_coll(self):
        return self.collection

    def __str__(self):
        return "Corpus: " + self.name + ", Number of docs: "+ str(self.ndoc)+ ", Number of authors: "+ str(self.naut)
    
    def __repr__(self):
        return self.name

    def sort_title(self,nreturn=None):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_title())][:(nreturn)]

    def sort_date(self,nreturn):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_date(), reverse=True)][:(nreturn)]
    
    def save(self,file):
            pickle.dump(self, open(file, "wb" ))
        
  #cette focntion retourne les passages des documents contenant le mot-clef entr´e
    def search(self,motif):
       motif=motif.lower()
       chaine=". ".join(str(doc) for doc in self.collection.values())
       chaine=self.nettoyer_texte(chaine)
       chaine = " ".join(str(x) for x in chaine)
       passage = re.findall(r"(?:^|\S+\s+\S+\s+\S+) {} (?:\s*\S+\s+\S+\s+\S+|$)".format(motif), chaine) 
       # l'extraction des passages a été fait se basant sur https://stackoverflow.com/questions/55255627/pythonhow-to-extract-a-word-before-and-after-the-match-using-regex
       if passage != []:
           return passage
       else:
           return("le mot ne figure pas dans le corpus")
       
    # cette fonction retroune le même résultat mais sous forme de dataframe
    def concorde(self,motif):
       motif=motif.lower()
       chaine=". ".join(str(doc) for doc in self.collection.values())
       chaine=self.nettoyer_texte(chaine)
       chaine = " ".join(str(x) for x in chaine)
       gauche=re.findall(r"(.*?:^|\S+\s+\S+\s+\S+) {} ".format(motif), chaine)
       droite=re.findall(r" {} (s*\S+\s+\S+\s+\S+|$)".format(motif), chaine)
       motif=[motif]*len(gauche)
       df = pd.DataFrame(list(zip(gauche, motif,droite)), 
               columns =['contexte gauche ', 'motif trouvé','contexte droit']) 
       return df
   
    
    #cette fonction gère tout le pré-traitement(tokennization)du corpus elle s'est essentiellement faite avec la librairie nltk 
    def nettoyer_texte(self,chaine):
        #enlever les url
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        chaine = url_pattern.sub(r'', chaine)
        #convert number
        chaine=re.sub("\d+", "", chaine)
        #split into words
        chaine=chaine.lower()
        chaine = word_tokenize(chaine)
        #remove punctuation
        chaine=[i.translate(str.maketrans("","", string.punctuation))for i in chaine]
       # remove remaining tokens that are not alphabetic
        chaine = [i for i in chaine if i.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        chaine = [i for i in chaine if not i in stop_words]
        # Lemmatizing of words
        chaine = [WordNetLemmatizer().lemmatize(i,pos="v") for i in chaine]
     
        return chaine
    
    #cette fonction contient le vocabulaire du corpus apès nettoyage 
    def vocabulaire(self):
        vocab=set()
        
        for doc in self.collection.values():
            a=doc.get_text()
            a= self.nettoyer_texte(str(a))
            for i in a:
                vocab.add(i)
        return vocab
    
    #cette fonction compte le nombre d’occurrences de chacun des mots de notre vocabulaire mais on ne l'a pas utilisé
    """def vocabulaireParDocument(self):
        
        vocab={}
        DF = defaultdict(int)
        for doc in self.collection.values():
            a=doc.get_text()
            a= self.nettoyer_texte(str(a))
            mot=re.findall(r'\w+',a)
            #b= a.split(" ")
            vocab["{}".format(doc.get_title())]=a
            for i in set(mot):
                if len(i) >= 3 and i.isalpha():
                    DF[i] += 1
            IDF = dict()
            for i in DF:
                IDF[i] = math.log(self.ndoc / float(DF[i]))
            #freq=pd.DataFrame(d.items(), columns=['Mot', 'Frequence'])
                #print(IDF)
        freq=pd.DataFrame()
        for cle, valeur in vocab.items():
            d=dict(Counter(re.findall(r'\w+', valeur)))
            freq=freq.append(pd.DataFrame(d.items(), columns=['Mot', 'term frequency']))
            #freq=freq.append(d, ignore_index=True)
        return freq#"""
    
    #Cette fonction génère à la fois le dataframe des co-occurences du corpus mais également son graphe
    #https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/calculate-tweet-word-bigrams/ ce lien nous a aidé pour les bigrams
    def co_occurence(self):
        # Create list of lists containing bigrams in tweets
        terms_bigram = [list(bigrams(self.nettoyer_texte(doc.get_text()))) for doc in self.collection.values() ]
      
        
        # Flatten list of bigrams in clean tweets
        bigra = list(itertools.chain(*terms_bigram))

        # Create counter of words in clean bigrams
        bigram_counts = Counter(bigra)
        bigram_df = pd.DataFrame(bigram_counts.items(),columns=['bigram', 'count'])
        self.result_df= bigram_df.head(self.N)

        df=bigram_df.head(self.N)
        
        df['bigram']=[",".join(i) for i in df['bigram']]
               
             
        df[['bigram','bigram2']] = df['bigram'].str.split(',',expand=True)
            
        df=df.rename(columns={'bigram': 'source','bigram2': 'target', 'count': 'label' })
            
        edges=df
            
        G = nx.from_pandas_edgelist(edges, edge_attr=True)
            

        G_Dyn = Network(height="750px", width="100%", bgcolor="white", font_color="#333333")
        G_Dyn.from_nx(G)
        voisin = G_Dyn.get_adj_list();
            
        for node in G_Dyn.nodes :
                #node["size"] =[ i for i in df['count'].values.tolist()]*len(voisin[node["id"]])
            node["size"] =20*len(voisin[node["id"]])
            node["color"] ="#048b9a"
            node["title"] ="co-occurents de "+ node["id"] +":<br>" + "<br>".join(voisin[node["id"]])
                
        for edges in G_Dyn.edges :
            edges["color"] ="#25FDE9"
            edges["width"] =10
            
        self.result = G_Dyn  
        return 
        
    #Cette fonction génère à la fois le dataframe des co-occurences d'un mot-clé du corpus mais également son graphe
    def co_occurenceMotif(self,motif):
                motif=motif.lower()
        
                text=self.search(motif)
                text = " ".join(str(x) for x in text)
                
                finder = BigramCollocationFinder.from_words(text.split())
                
                word_filter = lambda w1, w2: motif not in (w1, w2)
                finder.apply_ngram_filter(word_filter)
                        
                bigram_measures = nltk.collocations.BigramAssocMeasures()
                raw_freq_ranking = finder.nbest(bigram_measures.raw_freq, self.N) #top-10 vu sur https://stackoverflow.com/questions/49197667/nltk-how-to-get-bigrams-containing-a-specific-word
                
                        # Create counter of words in clean bigrams
                bigram_counts = Counter(raw_freq_ranking)
               
                bigram_df = pd.DataFrame(bigram_counts.items(),columns=['bigram', 'count'])
                
                self.result_df = bigram_df
                
                df=bigram_df
                
                df['bigram']=[",".join(i) for i in df['bigram']]
               
                df[['bigram','bigram2']] = df['bigram'].str.split(',',expand=True)
                df=df.rename(columns={'bigram': 'source','bigram2': 'target', 'count': 'label' })
            
                edges=df
            
                G = nx.from_pandas_edgelist(edges, edge_attr=True)
               
               # G = nx.from_pandas_edgelist(df,source="bigram",target="bigram2",label="count")
                
            
                #G = nx.from_pandas_edgelist(df,source="bigram1",target="bigram2")
                
                
                G_Dyn = Network(height="750px", width="100%", bgcolor="white", font_color="#333333")
                G_Dyn.from_nx(G)
                
                voisin = G_Dyn.get_adj_list();
                for node in G_Dyn.nodes :
                    node["title"] ="co-occurents de "+ node["id"] +":<br>" + "<br>".join(voisin[node["id"]])
                    if node["label"]== motif : 
                        
                        node["color"] = "#048b9a"
                        node["size"] =5*len(voisin[node["id"]])
                    else :
                        node["size"] =20
                        node["color"] = "#03224C"
                for edges in G_Dyn.edges :
                    
                    edges["color"] ="#25FDE9"
                    edges["width"] =3
                   
                self.result= G_Dyn
                return
            
    #Cette fonction affiche les centralités entre les valeurs propres du graphe d'un mot-clé donné , gérène le graphe et le sauvegarde sous format png
    def centralityMotif(self,motif):
            
            motif=motif.lower()
        
            text=self.search(motif)
            text = " ".join(str(x) for x in text)
                
            finder = BigramCollocationFinder.from_words(text.split())
                
            word_filter = lambda w1, w2: motif not in (w1, w2)
            finder.apply_ngram_filter(word_filter)
                        
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            raw_freq_ranking = finder.nbest(bigram_measures.raw_freq, self.N) #top-10
                
            # Create counter of words in clean bigrams
            bigram_counts = Counter(raw_freq_ranking)
               
            bigram_df = pd.DataFrame(bigram_counts.items(),columns=['bigram', 'count'])
                
            self.result_df = bigram_df
                
            df=bigram_df
                
            df['bigram']=[",".join(i) for i in df['bigram']]
               
            df[['bigram','bigram2']] = df['bigram'].str.split(',',expand=True)
            df=df.rename(columns={'bigram': 'source','bigram2': 'target', 'count': 'label' })
            
            edges=df
            
            G = nx.from_pandas_edgelist(edges, edge_attr=True)
           
            centrality=nx.eigenvector_centrality_numpy(G)
            
            
            for key, val in centrality.items():
                centrality[key]=round(val,2)
            
            pos=nx.spring_layout(G)
            groups = set(centrality.values())
            mapping = dict(zip(sorted(groups),count()))
            colors = [mapping[centrality.get(n,"none")] for n in centrality.keys()]
            nx.draw_networkx(G, pos, nodelist=centrality.keys(), node_color= colors, with_labels=True, 
                     node_size=[(v) * 3000 for v in centrality.values()])
            
            #fig, ax = plt.subplots()
            #plt.figure(figsize=(13,13))
            
            plt.title(label="Centralité des vecteurs propres \n\n"+str(centrality)+"\n", fontdict=None ,loc='center')
            plt.axis('off')
            #fig.tight_layout()
            plt.savefig("Graph.png", format="PNG")
            plt.show()
            
            
            
            
            
                                                                                                          
            
    #Cette fonction affiche les centralités entre les valeurs propres du graphe du corpus, gérène le graphe et le sauvegarde sous format png

    def centrality(self):
            
            # Create list of lists containing bigrams in tweets
            terms_bigram = [list(bigrams(self.nettoyer_texte(doc.get_text()))) for doc in self.collection.values() ]
          
            
            # Flatten list of bigrams in clean tweets
            bigra = list(itertools.chain(*terms_bigram))
    
            # Create counter of words in clean bigrams
            bigram_counts = Counter(bigra)
            bigram_df = pd.DataFrame(bigram_counts.items(),columns=['bigram', 'count'])
            
    
            df=bigram_df.head(self.N)
        
               
            df['bigram']=[",".join(i) for i in df['bigram']]
               
            df[['bigram','bigram2']] = df['bigram'].str.split(',',expand=True)
            df=df.rename(columns={'bigram': 'source','bigram2': 'target', 'count': 'label' })
            
            edges=df
            
            G = nx.from_pandas_edgelist(edges, edge_attr=True)
           
            centrality=nx.eigenvector_centrality_numpy(G)
            
            
            for key, val in centrality.items():
                centrality[key]=round(val,2)
            
            pos=nx.spring_layout(G)
            groups = set(centrality.values())
            mapping = dict(zip(sorted(groups),count()))
            colors = [mapping[centrality.get(n,"none")] for n in centrality.keys()]
            nx.draw_networkx(G, pos, nodelist=centrality.keys(), node_color= colors, with_labels=True, 
                     node_size=[(v) * 3000 for v in centrality.values()])
            
            
            plt.title(label="Centralité des vecteurs propres \n\n"+str(centrality)+"\n", fontdict=None ,loc='center')
            plt.axis('off')
         
            plt.savefig("Graph.png", format="PNG")
            plt.show()
            
                                                                                                          
           
    
    
            
    
                    
            

class Author():
    def __init__(self,name):
        self.name = name
        self.production = {}
        self.ndoc = 0
            
    def add(self, doc):     
        self.production[self.ndoc] = doc
        self.ndoc += 1

    def __str__(self):
        return "Auteur: " + self.name + ", Number of docs: "+ str(self.ndoc)
    def __repr__(self):
        return self.name
    


class Document():
    
    # constructor
    def __init__(self, date, title, author, text, url):
        self.date = date
        self.title = title
        self.author = author
        self.text = text
        self.url = url
        
  
    # getters
    
    def get_author(self):
        return self.author

    def get_title(self):
        return self.title
    
    def get_date(self):
        return self.date
    
    def get_source(self):
        return self.source
        
    def get_text(self):
        return self.text

    #def __str__(self):
       # return "Document " + self.getType() + " : " + self.title
    
    def __repr__(self):
        return self.title

    def sumup(self,ratio):
        try:
            auto_sum = summarize(self.text,ratio=ratio,split=True)
            out = " ".join(auto_sum)
        except:
            out =self.title            
        return out
    
    def getType(self):
        pass
  
# classe fille permettant de modéliser un Document Reddit
#

class RedditDocument(Document):
    
    def __init__(self, date, title,
                 author, text, url, num_comments):        
        Document.__init__(self, date, title, author, text, url)
        # ou : super(...)
        self.num_comments = num_comments
        self.source = "Reddit"
        
    def get_num_comments(self):
        return self.num_comments

    def getType(self):
        return "reddit"
    
    def __str__(self):
        #return(super().__str__(self) + " [" + self.num_comments + " commentaires]")
        return Document.__str__(self) + " [" + str(self.num_comments) + " commentaires]"
    
#
# classe fille permettant de modéliser un Document Arxiv
#

class ArxivDocument(Document):
    
    def __init__(self, date, title, author, text, url, coauteurs):
        #datet = dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
        Document.__init__(self, date, title, author, text, url)
        self.coauteurs = coauteurs
    
    def get_num_coauteurs(self):
        if self.coauteurs is None:
            return(0)
        return(len(self.coauteurs) - 1)

    def get_coauteurs(self):
        if self.coauteurs is None:
            return([])
        return(self.coauteurs)
        
    def getType(self):
        return "arxiv"

    def __str__(self):
        s = Document.__str__(self)
        if self.get_num_coauteurs() > 0:
            return s + " [" + str(self.get_num_coauteurs()) + " co-auteurs]"
        return s
    







