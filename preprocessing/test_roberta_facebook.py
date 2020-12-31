import torch
import esm
import numpy as np
# Load 34 layer model
# model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

batch_converter = alphabet.get_batch_converter()

# Prepare data (two protein sequences)
data = [("protein1", "MYLYQKIKN"), ("protein2", "MNAKYD")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

aa = alphabet.all_toks
# model.model_version
# Extract per-residue embeddings (on CPU)
with torch.no_grad():
    results = model(batch_tokens)
    pred = results['logits']
    prob = torch.softmax(pred,dim=2)
    pred_max = torch.argmax(prob,dim=2)
    pred_str = aa[pred_max[0,0,]]
    embedding = model(batch_tokens, repr_layers=[34])
    #
    # pred = model.lm_head(results)
token_embeddings = results["representations"][34]