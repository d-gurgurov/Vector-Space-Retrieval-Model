from bs4 import BeautifulSoup
from stemming.porter2 import stem
import string
import math


class SearchEngine:

    def __init__(self, collectionName, create):

        # the collectionName points to the filename of the document collection
        self.collectionName = collectionName
        # tf indexes
        self.tf = {}
        # idf indexes
        self.idf = {}

        # if create=True, the search index is created and written to files
        if create:

            # computing idf and tf scores
            self.compute_idf(collectionName)
            self.compute_tf(collectionName)

            # reading in the idf file
            with open(f"{collectionName}.idf", "r") as f:
                for line in f:
                    token, idf_value = line.strip().split("\t")
                    self.idf[token] = float(idf_value)

            # reading in the tf file
            with open(f"{collectionName}.tf", "r") as f:
                for line in f:
                    doc_id, token, tf_value = line.strip().split("\t")
                    tf_value = float(tf_value)
                    if doc_id in self.tf:
                        self.tf[doc_id][token] = tf_value
                    else:
                        self.tf[doc_id] = {token: tf_value}

        # if create=False, the search index is read from the files
        else:

            # reading in the idf file
            with open(f"{collectionName}.idf", "r") as f:
                for line in f:
                    token, idf_value = line.strip().split("\t")
                    self.idf[token] = float(idf_value)

            # reading in the tf file
            with open(f"{collectionName}.tf", "r") as f:
                for line in f:
                    doc_id, token, tf_value = line.strip().split("\t")
                    tf_value = float(tf_value)
                    if doc_id in self.tf:
                        self.tf[doc_id][token] = tf_value
                    else:
                        self.tf[doc_id] = {token: tf_value}


    def preprocess(self, collectionName):

        # reading in the file
        with open(f"{collectionName}.xml", 'r') as file:
            contents = file.read()

        # parsing the XML using BeautifulSoup
        soup = BeautifulSoup(contents, 'xml')

        # extracting the text from the HEADLINE and TEXT tags
        docs = []
        for doc in soup.find_all('DOC'):
            doc_id = doc['id']
            headline_tag = doc.find('HEADLINE')
            if headline_tag is not None:
                headline = headline_tag.text.lower().replace('\n', '')
            else:
                headline = ""
            headline_words = headline.split()
            headline_words = [token.translate(str.maketrans('', '', string.punctuation)) for token in
                              headline_words]
            headline_words = [stem(token) for token in headline_words if token]
            dateline = doc.find('DATELINE')
            if dateline is not None:
                dateline.extract()
            text = doc.find('TEXT').text.lower()
            tokens = text.split()
            # remove any punctuation characters from the tokens
            tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]
            # stemming the tokens
            tokens = [stem(token) for token in tokens if token]
            docs.append({'doc_id': doc_id, 'headline': headline, 'tokens': tokens + headline_words})

        return docs


    def compute_idf(self, collectionName):

        # extracting and pre-processing the file
        docs = self.preprocess(collectionName)

        # counting the number of documents containing each term
        doc_freq = {}
        for doc in docs:
            for token in set(doc['tokens']):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # computing the idf values for each term
        num_docs = len(docs)
        idf_values = {token: math.log(num_docs / freq) for token, freq in doc_freq.items()}

        # writing the idf values to a file
        with open(f"{collectionName}.idf", 'w') as file:
            for token, idf in sorted(idf_values.items()):
                file.write(str(token) + "\t" + str(idf) + "\n")


    def compute_tf(self, collectionName):

        # extracting and pre-processing the file
        docs = self.preprocess(collectionName)


        doc_tf = {}
        for doc in docs:
            doc_id = doc['doc_id']
            tokens = doc['tokens']
            # counting the number of occurrences of each token in the document
            token_counts = {}
            for token in tokens:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
            # finding the maximum count of any token in the document
            max_count = max(token_counts.values())
            # calculating the tf value for each token and adding it to the doc_tf dictionary
            for token, count in token_counts.items():
                tf = count / max_count
                if doc_id in doc_tf:
                    doc_tf[doc_id].append((token, tf))
                else:
                    doc_tf[doc_id] = [(token, tf)]

        # writing the tf values to a file
        with open(f"{collectionName}.tf", 'w') as file:
            for doc_id in sorted(doc_tf.keys()):
                tf_values = doc_tf[doc_id]
                for token, tf in sorted(tf_values):
                    file.write(str(doc_id) + "\t" + str(token) + "\t" + str(tf) + "\n")


    def executeQuery(self, queryTerms):

        # calculating the query vector (tf*idf)
        queryTerms = [stem(token.lower().translate(str.maketrans('', '', string.punctuation))) for token in queryTerms]
        query_vector = {}
        term_counts = {}

        for term in queryTerms:
            if term not in self.idf:
                continue
            if term in term_counts:
                term_counts[term] += 1
            else:
                term_counts[term] = 1
            max_count = max(term_counts.values())

            tf = queryTerms.count(term) / max_count
            idf = self.idf[term]
            query_vector[term] = tf * idf

        # calculating the document scores
        doc_scores = {}
        for doc_id, doc_tf in self.tf.items():
            doc_vector = {}
            for term, tf in doc_tf.items():
                if term not in self.idf:
                    continue
                idf = self.idf[term]
                doc_vector[term] = tf * idf
            dot_product = sum(query_vector.get(term, 0) * weight for term, weight in doc_vector.items())
            query_norm = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
            doc_norm = math.sqrt(sum(weight ** 2 for weight in doc_vector.values()))
            if query_norm == 0:
                cosine_sim = 0.0
            else:
                cosine_sim = dot_product / (query_norm * doc_norm)
            if cosine_sim != 0.0:
                doc_scores[doc_id] = cosine_sim

        # returning the top 10 documents (or less)

        return list(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10])


    def executeQueryConsole(self):

        while True:

            query = input("Please enter query, terms separated by whitespace: ")
            if not query:
                return "Bye!"

            # pre-processing the query terms
            query_terms = query.lower().split()
            query_terms = [stem(token.translate(str.maketrans('', '', string.punctuation))) for token in query_terms]
            # executing the query
            results = self.executeQuery(query_terms)[:10]
            # printing top 10 resutls (or less)
            if not results:
                print("Sorry, I didn’t find any documents for this query.")
            else:
                print("I found the following documents: ")
                for id, value in results:
                    print(f"{id} ({str(value)})")


if __name__ == '__main__':

    print("Reading (creating) index from file...")
    searchEngine = SearchEngine("nytsmall", create=True)
    print(searchEngine.executeQueryConsole())
