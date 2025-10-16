

import math
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from collections import namedtuple
from sacrebleu import corpus_bleu  # pip install sacrebleu


CHECKPOINT_PATH = "model_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"


class Vocab:
    def __init__(self, stoi: Dict[str,int], itos: Dict[int,str]):
        self.stoi = stoi
        self.itos = itos
    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi.get("<unk>", 0)) for t in tokens]
    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]


BeamHypothesis = namedtuple("BeamHypothesis", ["tokens", "log_prob", "state", "attn_weights"])


def beam_search_decode(encoder, decoder, src_tensor: torch.LongTensor, src_len: torch.LongTensor,
                       vocab: Vocab,
                       beam_size: int = 5,
                       max_len: int = 100,
                       length_penalty_alpha: float = 0.0,
                       n_best: int = 1,
                       early_stopping: bool = True):
    """
    encoder: encoder module, returns encoder_outputs, hidden (and optionally cell)
    decoder: decoder module with a method .step(input_token, hidden, encoder_outputs) -> (logits, next_hidden, attn)
             or you can adapt below to your decoder signature.
    src_tensor: (src_len,) or (1, src_len) LongTensor (already token-ids)
    src_len: scalar length tensor or int
    vocab: Vocab instance with stoi/itos
    Returns: list of best decoded token lists (n_best)
    """
    sos_id = vocab.stoi[SOS_TOKEN]
    eos_id = vocab.stoi[EOS_TOKEN]
    pad_id = vocab.stoi[PAD_TOKEN]

    # ---- run encoder ----
    encoder.eval(); decoder.eval()
    with torch.no_grad():
       
        encoder_outputs, encoder_hidden = encoder(src_tensor.unsqueeze(0).to(DEVICE), src_len.to(DEVICE))
        
        initial_state = encoder_hidden 
        beams = [BeamHypothesis(tokens=[sos_id], log_prob=0.0, state=initial_state, attn_weights=[])]

        completed_hyps: List[BeamHypothesis] = []

        for step in range(max_len):
            all_candidates: List[BeamHypothesis] = []

           
            if len(beams) == 0:
                break

            for beam in beams:
                last_token = beam.tokens[-1]
                if last_token == eos_id:
                    completed_hyps.append(beam)
                    continue

                input_token = torch.LongTensor([last_token]).to(DEVICE) 
                logits, next_state, attn = decoder.step(input_token, beam.state, encoder_outputs)
                log_probs = F.log_softmax(logits.squeeze(0), dim=-1) 

                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

                for k in range(topk_ids.size(0)):
                    token_id = int(topk_ids[k].item())
                    token_logprob = float(topk_log_probs[k].item())
                    new_tokens = beam.tokens + [token_id]
                    new_logprob = beam.log_prob + token_logprob
                    new_attn_weights = beam.attn_weights + [attn.cpu() if attn is not None else None]
                    new_state = next_state  
                    candidate = BeamHypothesis(tokens=new_tokens, log_prob=new_logprob, state=new_state, attn_weights=new_attn_weights)
                    all_candidates.append(candidate)

            if len(all_candidates) == 0:
                break

            def score(hyp: BeamHypothesis):
                lp = ((5.0 + len(hyp.tokens)) / 6.0) ** length_penalty_alpha if length_penalty_alpha > 0 else 1.0
                return hyp.log_prob / lp

            all_candidates.sort(key=score, reverse=True)
            beams = all_candidates[:beam_size]

            new_beams = []
            for b in beams:
                if b.tokens[-1] == eos_id:
                    completed_hyps.append(b)
                else:
                    new_beams.append(b)
            beams = new_beams

            if early_stopping and len(completed_hyps) >= n_best:
                break

        if len(completed_hyps) == 0:
            completed_hyps = beams

        completed_hyps.sort(key=score, reverse=True)
        best_hyps = completed_hyps[:n_best]

        decoded_sentences = []
        for hyp in best_hyps:
            tokens = hyp.tokens
            if tokens and tokens[0] == sos_id:
                tokens = tokens[1:]
            if eos_id in tokens:
                tokens = tokens[:tokens.index(eos_id)]
            decoded_tokens = vocab.decode(tokens)
            decoded_sentences.append((decoded_tokens, hyp.log_prob))

        return decoded_sentences
