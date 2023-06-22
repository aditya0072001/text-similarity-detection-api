from pymongo import MongoClient
import setting

client = MongoClient(setting.mongodb_uri, setting.port)
db = client['text_similarity']
similarities_collection = db["similarities"]

