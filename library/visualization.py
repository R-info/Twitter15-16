from typing import List, Tuple, Any
import umap
import json
import gensim
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis.gensim

from library import DS_PATH
from library.file_manager import save2csv, path_check_and_fix

COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "pink",
    "black",
    "orange",
    "purple",
    "beige",
    "brown",
    "gray",
    "cyan",
    "magenta"
]

def dimension_reductor(
    vectors,
    n_neighbors: int = 5,
    n_components: int = 2,
    min_dist: float = 0.0,
    metric: str = "correlation"
):
    ''' returns (n, 2) vectors '''
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        low_memory=True)
    reducer.fit(vectors[:min(50000, len(vectors))])

    return reducer

def tokenizer(text: str, ngram: str):
    if ngram == 'unigram':
        ntoken = 1
    elif ngram == 'bigram':
        ntoken = 2
    elif ngram == 'trigram':
        ntoken = 3
    else:
        ntoken = 1

    tokens = text.split(" ")

    if ntoken == 1:
        return tokens

    results = []
    token_len = len(tokens)
    for i in range(token_len):
        if i + ntoken > token_len:
            continue

        results.append("_".join(tokens[i:i + ntoken]))

    return results

def calculate_frequency(summary, topic_ids=None):
    all_data = {}
    for topic_id, info in summary['topics'].items():
        if topic_ids and topic_id not in topic_ids:
            continue
#         print(f"Topic : {topic_id}")
        for word in info['vocabs']:
            if word not in all_data:
                all_data[word] = 0
            all_data[word] += info['vocabs'][word]

    return sorted(all_data.items(), key=lambda item: item[1], reverse=True)

class TextsCluster:
    texts: List[Any]
    vis_coords: List[Tuple[float, float]]
    labels: List[int]
    summarypath: str = None

    def __init__(self, texts, vis_coords, labels):
        self.texts = [str(text) for text in texts]
        self.vis_coords = vis_coords
        self.labels = labels

    def records(self, ngram, filename = None):
        if ngram not in ['unigram', 'bigram', 'trigram']:
            print("N-Gram settings incorrect!")
            return

        print("Summary")
        summary = {}
        for lab in self.labels:
            if lab not in summary:
                summary[lab] = 0
            summary[lab] += 1
        for lab, val in sorted(summary.items(), key=lambda item: item[1], reverse=True):
            print(f"-- Topic {lab} : {round((val/len(self.labels))*100, 2)}%")

        print("Calculating Token Distribution")

        results = {
            "total_docs": 0,
            "topics": {}
        }
        for topic in set(self.labels):
            results['topics'][str(topic)] = {
                'count': 0,
                'vocabs': {}
            }

        for i, text in enumerate(self.texts):
            results['total_docs'] += 1
            results['topics'][str(self.labels[i])]['count'] += 1
            tokens = tokenizer(text, ngram)
            for token in set(tokens):
                if token not in results['topics'][str(self.labels[i])]['vocabs']:
                    results['topics'][str(self.labels[i])]['vocabs'][token] = 0
                results['topics'][str(self.labels[i])]['vocabs'][token] += 1

        if filename:
            self.summarypath = f"{DS_PATH}/data/processed/bow/summary/{filename}.json"
            filepath = path_check_and_fix(self.summarypath)
            with open(filepath, "w") as openfile:
                data = json.dump(results, openfile, indent=4)
            print(f"Saved at {self.summarypath}")

        return results

    def unigram_freq(self, filename = None):
        self.records("unigram", filename)
        return self

    def bigram_freq(self, filename = None):
        self.records("bigram", filename)
        return self

    def trigram_freq(self, filename = None):
        self.records("trigram", filename)
        return self

    def summary_report(self, filename: str = None, top_n: int = None):
        if not self.summarypath:
            print("Path to Token Summary Not Found")
            return

        print("loading summary...")
        with open(self.summarypath) as f:
            summary = json.load(f)

        print("Evaluate Words Frequency...")
        word_frequency = calculate_frequency(summary)
        csv_frequency = []
        for i in range(3):
            csv_frequency.append([])

        csv_frequency[0].append(f"Word Frequency")
        csv_frequency[1] += ["All Docs", ""]
        csv_frequency[2] += ["Word", "Freq"]
        for j, val in enumerate(word_frequency):
            if top_n and j >= top_n:
                break

            if len(csv_frequency) <= (j + 3):
                csv_frequency.append([])

            csv_frequency[j + 3] += [val[0], val[1]]

        for topic_id, info in summary['topics'].items():
            csv_frequency[1] += [f"Topic {topic_id}", ""]
            csv_frequency[2] += ["Word", "Freq"]
            for j, val in enumerate(sorted(info['vocabs'].items(), key=lambda item: int(item[1]), reverse=True)):
                if top_n and j >= top_n:
                    break

                if len(csv_frequency) <= (j + 3):
                    csv_frequency.append([])

                csv_frequency[j + 3] += [val[0], val[1]]

        if filename:
            filepath = f"{DS_PATH}/data/results/bow/summary/{filename}"
            filepath = path_check_and_fix(filepath)
            save2csv(csv_frequency, filepath)
            print(f"Saved at {filepath}")

    def visualize(self, filename: str = None, cmap: str = "brg_r"):
        result = pd.DataFrame(self.vis_coords, columns=['x', 'y'])
        result['labels'] = self.labels

        print("Visualizing Clusters...")
        fig, ax = plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(
            outliers.x,
            outliers.y,
            color='#BDBDBD',
            s=0.05
        )
        plt.scatter(
            clustered.x,
            clustered.y,
            c=clustered.labels,
            s=0.05,
            cmap=cmap
        )
        plt.colorbar()
        print()

        if filename:
            filepath = f"{DS_PATH}/data/results/visualization/clustering/{filename}"
            filepath = path_check_and_fix(filepath)
            plt.savefig(filepath)
            print(f"Saved at {filepath}")

class LdaData:
    model = None
    texts: List[List[str]]
    n_topics: int

    coherence_score: float = None
    perplexity_score: float = None

    def __init__(self, texts: List[str], n_topics: int, model=None):
        self.model = model
        self.n_topics = n_topics
        self.texts = [str(text).split(" ") for text in texts]

        bow_corpus = [self.model.id2word.doc2bow(doc) for doc in self.texts]
        tfidf = gensim.models.TfidfModel(bow_corpus)
        self.corpus = tfidf[bow_corpus]

    def evaluate(self):
        coherence_model_lda = gensim.models.CoherenceModel(
            model=self.model,
            texts=self.texts,
            dictionary=self.model.id2word,
            coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        print('-- Perplexity: ', self.model.log_perplexity(self.corpus))
        print('-- Coherence Score: ', coherence_lda)
        return self

    def get_ldavis(self, filename):
        print("Generating LDAvis...")
        vis = pyLDAvis.gensim.prepare(
            self.model,
            self.corpus,
            dictionary=self.model.id2word,
            mds='mmds')

        print("saving visualization...")
        filepath = f"{DS_PATH}/data/results/visualization/LDAvis/{filename}.html"
        filepath = path_check_and_fix(filepath)
        pyLDAvis.save_html(vis, filepath)
        print(f"Saved at {filepath}")
