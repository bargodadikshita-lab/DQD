import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

class DuplicateDetector:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.questions = self.data['question'].tolist()
        self.answers = self.data['answer'].tolist()

        embedding_path = "data/embeddings.pt"

        if os.path.exists(embedding_path):
            print("Loading embeddings...")
            self.embeddings = torch.load(embedding_path)
        else:
            print("Creating embeddings...")
            self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)
            torch.save(self.embeddings, embedding_path)

    def find_similar(self, query, threshold=0.5, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        scores = util.cos_sim(query_embedding, self.embeddings)[0]

        print("Similarity Scores:", scores)  # DEBUG

        top_results = torch.topk(scores, k=top_k)

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "question": self.questions[idx],
                "answer": self.answers[idx],
                "score": float(score)
            })

        best_score = float(top_results.values[0])

        return {
            "duplicate": best_score >= threshold,
            "best_score": best_score,
            "matches": results
        }

    def add_new_qa(self, question, answer, path):
        new_row = pd.DataFrame([[question, answer]], columns=["question", "answer"])
        new_row.to_csv(path, mode='a', header=False, index=False)