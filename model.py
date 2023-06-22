from sentence_transformers import SentenceTransformer, util

query = "DevOps along wtih ChatGPT reference"
docs = ["An open-source project called Docker (not the ship ones..) automates ( like most people are automating coding with ChatGPT) the deployment of applications inside software containers, packages that contain all the components necessary for an application to execute, such as code, runtime, system tools, and libraries, but not other programs dependencies, such as configuration files or libraries.", "In order to automate and streamline the software development lifecycle, DevOps is a collaborative methodology that unites teams ( even if you are not on good terms with them) from the development, testing, and operations departments."]


model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

query_emb = model.encode(query)
doc_emb = model.encode(docs)


scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()


doc_score_pairs = list(zip(docs, scores))


doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)


print("Query:", query)

for doc, score in doc_score_pairs:
    print(score," :: ", doc)
