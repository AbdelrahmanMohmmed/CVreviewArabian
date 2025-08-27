import opendatasets as od

od.download(
    "https://www.kaggle.com/datasets/akshatkjain/job-postings",
    data_dir="data/job_postings"
)
