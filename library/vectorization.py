import gensim
import torch
import torch.nn as nn
import numpy as np
from gensim import corpora, models
from sentence_transformers import SentenceTransformer
from typing import List
from torch.autograd import Variable
from library import DS_PATH

model_list = [
    "distilbert-base-nli-mean-tokens",  # english only
    "stsb-xlm-r-multilingual"   # bahasa indonesia included
]

def get_SBERT_vectorized(
    data: List[str],
    model_name: str = "distilbert-base-nli-mean-tokens"
) -> List[List[float]]:
    print(f'''Getting vector representations for SBERT using "{model_name}" ...''')
    model = SentenceTransformer(model_name)
    vec = np.array(model.encode(data, show_progress_bar=False))

    print('Getting vector representations for SBERT. Done!')
    return vec

def get_LDA_vectorized(data: List[List[str]], lda_model = None, debug = True):
    if not lda_model:
        print("Please include LDA Model")
        return

    if debug: print('Getting vector representations for LDA ...')
    def get_vec_lda(model, corpus, k):
        """
        Get the LDA vector representation (probabilistic topic assignments for all documents)
        :return: vec_lda with dimension: (n_doc * n_topic)
        """
        n_doc = len(corpus)
        vec_lda = np.zeros((n_doc, n_topics))
        for i in range(n_doc):
            # get the distribution for the i-th document in corpus
            for topic, prob in model.get_document_topics(corpus[i]):
                vec_lda[i, topic] = prob

        return vec_lda

    if debug: print("-- calculating Corpus")
    bow_corpus = [lda_model.id2word.doc2bow(doc) for doc in data]
    tfidf = models.TfidfModel(bow_corpus)
    corpus = tfidf[bow_corpus]

    n_topics = 0
    for i, topic in lda_model.print_topics(-1):
        if debug: print(topic)
        n_topics += 1

    vec = get_vec_lda(lda_model, corpus, n_topics)
    if debug: print('Getting vector representations for LDA. Done!')
    return vec

def concat_lda_sbert(lda_vec, sbert_vec):
    return np.c_[lda_vec * 15, sbert_vec]

class Autoencoder(nn.Module):

    def __init__(self, input_vec):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_vec, 32),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(32, input_vec),
            nn.ReLU(True))

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_autoencoder(data):
    # print("Check GPU Availability")
    # print(K.tensorflow_backend._get_available_gpus())

    ae_model = Autoencoder(data.shape[1]).cuda()

    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_model.parameters(),lr=0.0001)

    print('Fitting Autoencoder ...')
    num_epochs = 20
    data = torch.from_numpy(data).float()
    for epoch in range(num_epochs):
        for vect in data[:min(50000, len(data))]:
            vect = Variable(vect).cuda()
            # ===================forward=====================
            output = ae_model(vect)
            loss = distance(output, vect)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
    print('Fitting Autoencoder Done!')

#     return ae_model.encoder(data.cuda()).cpu().detach().numpy()
    return ae_model

def append_vector(vector, filepath: str):
    print("Saving Vectors...")
    with open(f"{DS_PATH}/{filepath}", "ab") as f:
        np.savetxt(f, vector, delimiter=',', fmt='%s')

def concatenate_autoencode(lda_vect, sbert_vec, filepath: str):
    vector = concat_lda_sbert(lda_vect, sbert_vec)

    autoencoder_model = get_autoencoder(vector)

    vector = autoencoder_model.encoder(torch.from_numpy(vector).float().cuda()).cpu().detach().numpy()
    append_vector(vector, filepath)

def sbert_vectorize_and_save(
    data: List[str],
    sbert_model: str = None,
    savepath: str = None
) -> List[List[float]]:
    if not sbert_model:
        print("Please Specify SBERT Model name!")
        return

    print(f"Data length : {len(data)}")
    print("Getting SBERT Vectors")
    vectors = get_SBERT_vectorized(data, sbert_model)
    print(f"Vectors shape : {vectors.shape}")

    if savepath:
        print("-- savings SBERT vec...")
        with open(f"{DS_PATH}/{savepath}", "wb") as f:
            np.savetxt(f, vectors, delimiter=',', fmt='%s')
    else:
        print("Not Savings vectors")

    return vectors

def lda_vectorize_and_save(
    data: List[List[str]],
    lda_model = None,
    savepath: str = None
) -> List[List[float]]:
    if not lda_model:
        print("Please Specify LDA Model!")
        return

    print(f"Data length : {len(data)}")
    print("Getting LDA Vectors")
    vectors = get_LDA_vectorized(data, lda_model)
    print(f"Vectors shape : {vectors.shape}")

    if savepath:
        print("-- savings LDA vec...")
        with open(f"{DS_PATH}/{savepath}", "wb") as f:
            np.savetxt(f, vectors, delimiter=',', fmt='%s')
    else:
        print("Not Savings vectors")

    return vectors
