from pymilvus import connections


if __name__ == '__main__':
    import os
    import pickle
    CONNECTION_URI = os.getenv("CONNECTION_URI")
    connections.connect(uri=CONNECTION_URI)
    with open("barkin.pkl", "rb") as f:
        vector_store = pickle.load(f)
    vectordb_docs = vector_store.similarity_search("Veri Sorumluları Siciline kayıt yükümlülüğüne istisna getirilen veri sorumluları kimlerdir?")
    print(vectordb_docs)
    