import re
import json


DATA_FILE = "data/train.json"

def main():
    with open(DATA_FILE, "r") as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents")


    # There are two types of questions in the data set. Type 2 questions have multiple questions in the same document.
    type_2_documents = [doc for doc in documents if 'qa_1' in doc]
    type_1_documents = [doc for doc in documents if 'qa_1' not in doc]

    print("Num Type 1 questions:", len(type_1_documents))
    print("Num Type 2 questions:", len(type_2_documents))

    def is_percentage_or_numeric(value):
        """
        Check if the value is numeric (including decimals and negatives) or ends with a percentage sign.
        """
        pattern = r"^-?\d+(\.\d+)?%?$"
        return isinstance(value, str) and re.match(pattern, value)



    for document in type_1_documents:

        assert is_percentage_or_numeric(document["qa"]["answer"]), f"Answer {document['qa']['answer']} is invalid"

    for document in type_2_documents:
        assert is_percentage_or_numeric(document["qa_0"]["answer"]), f"Answer {document['qa_0']['answer']} is invalid"
        assert is_percentage_or_numeric(document["qa_1"]["answer"]), f"Answer {document['qa_1']['answer']} is invalid"

if __name__ == "__main__":
    main()
