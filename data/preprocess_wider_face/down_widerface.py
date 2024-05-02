from huggingface_hub import snapshot_download
# import os

# os.environ['HF_DATASETS_CACHE']='./'
snapshot_download(repo_id="wider_face", repo_type="dataset", local_dir='./')