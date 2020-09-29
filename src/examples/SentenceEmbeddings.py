from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

'''
roberta-large-nli-stsb-mean-tokens - STSb performance: 86.39
roberta-base-nli-stsb-mean-tokens - STSb performance: 85.44
bert-large-nli-stsb-mean-tokens - STSb performance: 85.29
distilbert-base-nli-stsb-mean-tokens - STSb performance: 85.16

bert-base-wikipedia-sections-mean-tokens

average_word_embeddings_glove.6B.300d
average_word_embeddings_komninos
average_word_embeddings_levy_dependency
average_word_embeddings_glove.840B.300d

'''