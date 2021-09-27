import gensim
import os
from gensim import corpora, models
from typing import Dict, List
from library import DS_PATH
from library.vectorization import get_LDA_vectorized
from library.clustering import get_topics_label


def lda_training(
    data: List[List[str]],
    params: Dict = {},
    save_model: str = None
):
    '''
    data must already be preprocessed,
    params : {n_topics, alpha, eta},
    save_model : filename to save the model
    '''
    print('Training LDA model...')
    n_topics = params.get('n_topics', 16)
    alpha = params.get('alpha', 'auto')
    eta = params.get('eta', None)
    no_below = params.get('no_below', 5)
    no_above = params.get('no_above', 0.7)
    passes = params.get('passes', 5)
    chunksize = params.get('chunksize', min(len(data), 100_000))

    if params:
        print("Params :")
        for k, v in params.items():
            print(f"- {k} : {v}")

    print(f"-- calculating Corpus")

    dictionary = gensim.corpora.Dictionary(data)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)

    bow_corpus = [dictionary.doc2bow(doc) for doc in data]
    tfidf = models.TfidfModel(bow_corpus)
    corpus = tfidf[bow_corpus]

    print("-- training LDA Model...")
    ldamodel = gensim.models.ldamodel.LdaModel(
    # ldamodel = gensim.models.ldamulticore.LdaMulticore(
    #     workers=3,
        corpus=corpus,
        num_topics=n_topics,
        id2word=dictionary,
        passes=passes,
        random_state=100,
        alpha=alpha,
        eta=eta,
        chunksize=chunksize,
        per_word_topics=True)

    print(ldamodel)

    # print(f"-- alpha : {ldamodel.alpha}")
    # print(f"-- eta : {ldamodel.eta}")

    coherence_model_lda = gensim.models.CoherenceModel(model=ldamodel, texts=data, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    lda_vectors = get_LDA_vectorized(data, ldamodel, debug = False)
    labels = get_topics_label(lda_vectors)
    labels_count = len(set(labels))

    print('Perplexity: ', ldamodel.log_perplexity(corpus))
    print('Coherence Score: ', coherence_lda)
    print('Dataset Count: ', len(labels))
    print(f"Topic Coverage : {labels_count}/{n_topics}")
    print()

    if save_model:
        current_path = os.path.abspath(__file__)
        print(f"-- saving LDA models\n")
        ldamodel.save(f"{DS_PATH}/data/models/LDA/{save_model}.model")


def multi_topics_lda(
    data: List[List[str]],
    n_topics: List[int],
    params: Dict = {},
    save_model: str = None
):
    '''
    data must already be preprocessed,
    n_topics: list of n_topic in integer,
    params : {n_topics, alpha, eta},
    save_model : filename to save the model
    '''
    for n_topic in n_topics:
        model_filename = None if not save_model else f"{save_model}_{n_topic}"
        params['n_topics'] = n_topic
        lda_training(data, params, model_filename)
