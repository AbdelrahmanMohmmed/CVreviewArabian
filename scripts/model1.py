import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
import PyPDF2

df = pd.read_csv('/home/jax/CVreviewArabian/data/preprocessed/sampled_cleaned.csv')


df['text'] = df["Job Description"] + " " + df["skills"] + " " + df["Responsibilities"]
le = LabelEncoder()
df['label'] = le.fit_transform(df['Job Title'])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# ----------------------------
# 4️⃣ تحويل البيانات لـ Dataset PyTorch
# ----------------------------
class JobDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = JobDataset(train_encodings, train_labels)
test_dataset = JobDataset(test_encodings, test_labels)

# ----------------------------
# 5️⃣ تهيئة الموديل للتصنيف
# ----------------------------
num_labels = len(df['label'].unique())
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# ----------------------------
# 6️⃣ إعداد التدريب
# ----------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

from sklearn.metrics import accuracy_score

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ----------------------------
# 7️⃣ تدريب الموديل
# ----------------------------
trainer.train()

# ----------------------------
# 8️⃣ حفظ الموديل بعد التدريب
# ----------------------------
model.save_pretrained("trained_job_model")
tokenizer.save_pretrained("trained_job_model")

# ----------------------------
# 9️⃣ دالة لقراءة PDF واستخراج النص
# ----------------------------
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# ----------------------------
# 10️⃣ دالة لتحديد المهارات الناقصة
# ----------------------------
def missing_skills(cv_text, required_text):
    cv_skills = set(cv_text.lower().split())
    required_skills = set(required_text.lower().split())
    missing = required_skills - cv_skills
    return list(missing)

# ----------------------------
# 11️⃣ دالة للتنبؤ باستخدام الموديل المحفوظ
# ----------------------------
# إعادة تحميل الموديل المحفوظ
tokenizer = AutoTokenizer.from_pretrained("trained_job_model")
model = AutoModelForSequenceClassification.from_pretrained("trained_job_model")
model.eval()  # وضع الموديل في وضع التقييم

def predict_job_with_skills(pdf_path):
    cv_text = read_pdf(pdf_path)
    inputs = tokenizer(cv_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
    best_idx = probs.argmax()
    best_category = le.inverse_transform([best_idx])[0]
    best_prob = probs[best_idx] * 100
    required_text = df[df['label']==best_idx]['full_requirements'].values[0]
    missing = missing_skills(cv_text, required_text)
    return best_category, best_prob, missing

# ----------------------------
# 12️⃣ تجربة على ملف PDF
# ----------------------------
pdf_file = "/home/jax/CVreviewArabian/data/testCV/DataEngineerCv_V1.pdf"
category, probability, missing = predict_job_with_skills(pdf_file)
print(f"Predicted Job Category: {category}")
print(f"Probability: {probability:.2f}%")
print(f"Missing Skills: {missing}")