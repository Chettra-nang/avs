I. Create catch LLM and Text embeddings 
  - In main folder 
  1.1. Run main/0_build_manifests.ipynb to get captions and manifests 
  1.2. Run main/make_text_embeddings.ipynb to get cached_llm 


Current situation: 

+ Train one PPO policy for the ego ambulance only.

+ All other cars are the stock highway-env vehicles (IDM). They don’t know the ego is an ambulance and won’t deliberately yield. They just follow car-following rules and avoid collisions.

+ The CLIP/Text bits only help the ego’s state representation; they don’t change the behavior of other agents.





# from your repo root (so tools/ exists)
PY=python

DS_ROOT="/Users/nginkimlong/Documents/PHD/Exchange Program (SEED)/PROJECTs/ambulance_dataset_15k_cpu"
OUT_DIR="/Users/nginkimlong/Documents/PHD/Exchange Program (SEED)/Ambulance_EGO/cached_llm"

$PY tools/make_text_embeddings.py \
  --dataset-root "$DS_ROOT" \
  --out-dir "$OUT_DIR" \
  --provider local \
  --local-model "sentence-transformers/all-MiniLM-L6-v2" \
  --write-per batch
