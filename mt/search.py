import numpy as np


def get_top_tokens(model, args, trg_indexes, beam_width, eos_idx, max_trg_len):
    # Get top k words per beam/sentence
    top_k_beam = []
    attentions = []
    for i in range(len(trg_indexes)):
        _idxs = trg_indexes[i][0]

        # Check if the EOS has been reached, or the maximum length exceeded
        if _idxs[-1] == eos_idx or len(_idxs) >= max_trg_len:
            top_k_beam.append(None)
            attentions.append(None)
        else:
            # Get top tokens
            (probs, idxs), last_attention = model.decode_word(*args, trg_indexes=_idxs)
            probs, idxs = probs[0, :beam_width], idxs[0, :beam_width]  # 1 batch, get top k (we don't need more)
            probs, idxs = probs.cpu(), idxs.cpu()  # k preds to CPU

            # Add results
            top_k_beam.append((idxs, probs))
            attentions.append(last_attention)
    return top_k_beam, attentions


def update_top_k(trg_indexes, new_sentences, beam_width, eos_idx):
    def prob_sentence(x):
        idxs, probs, attn = x
        p_norm = float(1 / len(probs) * np.array(x[1]).prod())  # Normalize by length
        p_norm += 10e8 if idxs[-1] == eos_idx else 0  # Finished sentences have priority
        p_norm += len(idxs)  # Longer sentences have priority
        p_norm += 0 if attn is None else 10e-8  # Add epsilon in case there is attention (avoiding problems)
        return p_norm

    new_trg_indexes = trg_indexes + new_sentences  # Add new sentences
    new_trg_indexes.sort(key=lambda x: prob_sentence(x), reverse=True)  # sort by prob
    new_trg_indexes = new_trg_indexes[:beam_width]  # Get top predictions
    return new_trg_indexes


def greedy_search(trg_indexes, top_k_beam, attention=None):
    new_idxs = trg_indexes[0][0] + [int(top_k_beam[0][0])]
    new_probs = trg_indexes[0][1] + [float(top_k_beam[0][1])]
    attn = attention[0] if attention else None

    return [(new_idxs, new_probs, attn)]


def beam_search(trg_indexes, top_k_beam, beam_width, attention=None):
    # Get tuples
    tuples = []
    for i in range(len(top_k_beam)):  # num. beams
        if top_k_beam[i] is not None:
            _idxs, _probs = top_k_beam[i]

            for j in range(len(_idxs)):  # candidates
                idx = int(_idxs[j])
                prob = float(_probs[j])
                tuples.append((i, idx, prob))  # sent_i, prob, vocab_idx

    # Get top k sentences
    tuples.sort(key=lambda x: x[2], reverse=True)
    tuples = tuples[:beam_width] if tuples else tuples

    # Compute new sentences
    new_sentences = []
    for sent_i, idx, prob, in tuples:
        new_idxs = trg_indexes[sent_i][0] + [idx]
        new_probs = trg_indexes[sent_i][1] + [prob]
        new_attn = attention[sent_i] if attention else None
        new_sentences.append((new_idxs, new_probs, new_attn))

    return new_sentences
