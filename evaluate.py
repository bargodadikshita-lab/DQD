from main import DuplicateDetector

detector = DuplicateDetector("data/qa_dataset.csv")

questions = [
    "Explain Python",
    "What is ML",
    "How to lose fat",
    "What is quantum computing"
]

for q in questions:
    result = detector.find_similar(q)
    print(q, "->", result["best_score"])