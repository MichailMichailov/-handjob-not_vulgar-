import os
# from dotenv import load_dotenv

# load_dotenv()

parent_dir = os.path.dirname(os.path.abspath(__file__))
resultFile = os.path.join(parent_dir, '', "result.png")
train_data_folder = os.path.join(parent_dir, '', "knn")

emnist_labels = {
    **{i: str(i) for i in range(10)},  # 0-9 (цифры)
    **{i + 10: chr(65 + i) for i in range(26)},  # 10-35 (A-Z)
    **{i + 36: chr(97 + i) for i in range(26)}  # 36-61 (a-z)
}