from fastapi import FastAPI, status, File, UploadFile
from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import database
import PyPDF2
import os
import textractplus as tp
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
import traceback
import cleantxty
import cleantext

class Similarities(BaseModel):
    """
    Represents the data model for similarity entries.

    Attributes:
        original_text (str): The original text for comparison.
        similar_texts (list): List of similar texts.
    """

    original_text: str
    similar_texts: list

    def __str__(self):
        """
        Returns the string representation of the Similarities object.
        """
        return "%s %s %s" % (self.original_text, self.summary, self.keywords)


def similarity_model(text, docs):
    """
    Performs similarity modeling between a query text and a list of document texts.

    Args:
        text (str): The query text.
        docs (list): List of document texts.

    Returns:
        list: Sorted list of document-score pairs based on similarity scores.
    """

    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

    query_emb = model.encode(text)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = list(zip(docs, scores))

    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    return doc_score_pairs


def pdf_text_extraction(file):
    """
    Extracts text from a PDF file.

    Args:
        file (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """

    pdf_text = ''
    pdfFileObj = open(file, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    for i in range(0, len(pdfReader.pages)):
        pageObj = pdfReader.pages[i]
        pdf_text = pdf_text + pageObj.extract_text()

    pdfFileObj.close()

    return pdf_text


similarity_api = FastAPI()

@similarity_api.post("/check_similarities/", status_code=status.HTTP_201_CREATED)
async def create_similarity(similarity: Similarities):
    """
    Creates a similarity entry by comparing the original text with a list of similar texts.

    Args:
        similarity (Similarities): The similarity data containing original text and similar texts.

    Returns:
        dict: Similarity results.
    """

    similarity_exists = False
    similarity_dict = similarity.dict()

    try:
        if database.similarities_collection.count_documents({'original_text': similarity_dict['original_text']}) > 0:
            similarity_exists = True
            data = database.similarities_collection.find_one({'original_text': similarity_dict['original_text']})
            return data['similar_texts']

        elif not similarity_exists:
            results = similarity_model(similarity_dict['original_text'], similarity_dict['similar_texts'])
            similarity_dict.update({"similar_texts": results})
            result_db = database.similarities_collection.insert_one(similarity_dict)
            ack = result_db.acknowledged
            return {'similar_texts': results}

    except Exception as e:
        return {"message": "Error Occurred", "error": str(e)}


@similarity_api.post("/check_similarities_pdf/", status_code=status.HTTP_201_CREATED)
async def create_similarity_pdf(files: List[UploadFile] = File(...)):
    """
    Creates a similarity entry by comparing the text extracted from PDF files.

    Args:
        files (List[UploadFile]): List of PDF files.

    Returns:
        dict: Similarity results.
    """

    try:
        similarity_dict = {}
        similarity_exists = False

        if len(files) == 0:
            return {"message": "No files uploaded."}

        for file in files:
            suffix = Path(file.filename).suffix
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)

            # Check if the file is empty
            if tmp_path.stat().st_size == 0:
                return {"message": f"The file '{file.filename}' is empty."}

            # Reset file position to the beginning
            file.file.seek(0)

            pdf_text = pdf_text_extraction(str(tmp_path))
            similarity_dict['original_text'] = pdf_text

            if database.similarities_collection.count_documents({'original_text': pdf_text}) > 0:
                similarity_exists = True
                data = database.similarities_collection.find_one({'original_text': pdf_text})
                return data['similar_texts']

        if not similarity_exists:
            similarity_results = {}
            for file in files:
                suffix = Path(file.filename).suffix
                with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    tmp_path = Path(tmp.name)

                # Check if the file is empty
                if tmp_path.stat().st_size == 0:
                    return {"message": f"The file '{file.filename}' is empty."}

                # Reset file position to the beginning
                file.file.seek(0)

                pdf_text = pdf_text_extraction(str(tmp_path))
                similarity_results[file.filename] = similarity_model(pdf_text, similarity_dict['original_text'])

            similarity_dict['similar_texts'] = similarity_results
            result_db = database.similarities_collection.insert_one(similarity_dict)
            ack = result_db.acknowledged

            return {'similar_texts': similarity_results}

    except Exception as e:
        error_traceback = traceback.format_exc()
        return {"message": "An error occurred", "error": str(e), "traceback": error_traceback}


from bson import ObjectId

@similarity_api.get("/similarities/")
async def get_similarities():
    """
    Retrieves all similarity entries from the database.

    Returns:
        list: List of similarity entries.
    """

    try:
        similarities = []
        for similarity in database.similarities_collection.find():
            similarity['_id'] = str(similarity['_id'])  # Convert ObjectId to string
            similarities.append(similarity)
        return similarities
    except Exception as e:
        return {"message": "An error occurred", "error": str(e)}


@similarity_api.get("/similarities/{id}")
async def get_similarity(id: str):
    """
    Retrieves a specific similarity entry by its ID.

    Args:
        id (str): The ID of the similarity entry.

    Returns:
        dict: The similarity entry.
    """

    try:
        similarity = database.similarities_collection.find_one({'_id': ObjectId(id)})  # Convert string to ObjectId
        if similarity:
            similarity['_id'] = str(similarity['_id'])  # Convert ObjectId to string
            return similarity
        else:
            return {"message": "No similarity found with the given ID"}
    except Exception as e:
        return {"message": "An error occurred", "error": str(e)}


@similarity_api.post("/check_similarities_files/", status_code=status.HTTP_201_CREATED)
async def create_similarity_files(files: List[UploadFile] = File(...)):
    """
    Creates a similarity entry by comparing the text extracted from multiple files.

    Args:
        files (List[UploadFile]): List of files to compare.

    Returns:
        dict: Similarity results.
    """

    try:
        similarity_dict = {}
        similarity_exists = False

        if len(files) == 0:
            return {"message": "No files uploaded."}

        for file in files:
            suffix = Path(file.filename).suffix
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:  # Open file in binary mode
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)

            # Check if the file is empty
            if tmp_path.stat().st_size == 0:
                return {"message": f"The file '{file.filename}' is empty."}

            # Reset file position to the beginning
            file.file.seek(0)

            text = tp.process(str(tmp_path))
            text = cleantxty.clean(str(text))
            print(text)   
            similarity_dict['original_text'] = text

            if database.similarities_collection.count_documents({'original_text': text}) > 0:
                similarity_exists = True
                data = database.similarities_collection.find_one({'original_text': text})
                return data['similar_texts']

        if not similarity_exists:
            similarity_results = {}
            for file in files:
                suffix = Path(file.filename).suffix
                with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:  # Open file in binary mode
                    shutil.copyfileobj(file.file, tmp)
                    tmp_path = Path(tmp.name)

                # Check the file size
                file_size = os.path.getsize(tmp.name)

                if file_size == 0:
                    return {"message": f"Empty file: {file.filename}"}

                # Reset file position to the beginning
                file.file.seek(0)

                text = tp.process(str(tmp_path))
#                text = cleantext.clean(str(text))
                text = cleantxty.clean(str(text))
                print("text { ", text,"'end' }")
                similarity_results[file.filename] = similarity_model(text, similarity_dict['original_text'])

            similarity_dict['similar_texts'] = similarity_results
            result_db = database.similarities_collection.insert_one(similarity_dict)
            ack = result_db.acknowledged

            return {'similar_texts': similarity_results}

    except Exception as e:
        error_traceback = traceback.format_exc()
        return {"message": "An error occurred", "error": str(e), "traceback": error_traceback}


@similarity_api.get("/")
async def root():
    """
    Default endpoint.

    Returns:
        dict: Welcome message.
    """

    return {"message": "Welcome to the Text Similarity Detection API"}
