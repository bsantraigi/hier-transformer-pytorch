"""
Copyright (c) 2021 by Bishal Santra

All rights reserved.
This file is part of the hier-transformer,
and is released under the "MIT License Agreement". Please see the LICENSE
file that should have been included as part of this package.
"""

import torch


# The dimensions and order of axes needs to be LxS for all masks.
# Pytorch transformer.py expects attn_masks to be LxS
# where L is the target sequence length, S is the source sequence length

def gen_encoder_ut_mask(src_seq, input_mask, utt_loc):
    def _gen_mask_hierarchical(A, src_pad_mask):
        # A: (bs, 100, 100); 100 is max_len*2 same as input_ids
        return ~(2 * A == (A + A.transpose(1, 2))).bool()
    enc_mask_utt = _gen_mask_hierarchical(utt_loc.unsqueeze(1).expand(-1, src_seq.shape[1], -1), input_mask)
    return enc_mask_utt

def _get_pe_inputs(tgt_seq, src_seq, input_mask, utt_loc):
    pe_utt_loc = torch.zeros(utt_loc.shape, device=utt_loc.device)
    for i in range(1, utt_loc.shape[1]):  # time
        _logic = (utt_loc[:, i] == utt_loc[:, i - 1]).float()
        pe_utt_loc[:, i] = pe_utt_loc[:, i - 1] + _logic - (1 - _logic) * pe_utt_loc[:, i - 1]
    return pe_utt_loc


def _HIER_masks(tgt_seq, src_seq, input_mask, utt_loc):
    # HT-Encoder
    pe_utt_loc = _get_pe_inputs(tgt_seq, src_seq, input_mask, utt_loc)

    # UT-MASK
    enc_mask_utt = gen_encoder_ut_mask(src_seq, input_mask, utt_loc)

    # CT-Mask HIER style
    # Note: The sequence is all utterance followed by kb entries
    # We attend to the final *utterance*
    _x = (utt_loc == (utt_loc.max(1, keepdim=True).values)).unsqueeze(1)
    final_utt_check = _x.expand(-1, src_seq.shape[1], -1)
    #     print(utt_loc[0:5])
    #     print(final_utt_check[0:5, 0, :]*1)

    enc_mask_ct = (~((~enc_mask_utt) | final_utt_check.transpose(1, 2)))  # Real HRED

    # For HIER style
    # dec_enc_attn_mask = input_mask.unsqueeze(1).expand(-1, tgt_seq.shape[1], -1)
    dec_enc_attn_mask = (~_x).expand(-1, tgt_seq.shape[1], -1)
    # print(dec_enc_attn_mask.shape)
    return pe_utt_loc, enc_mask_utt, enc_mask_ct, dec_enc_attn_mask


def _CLS_masks(tgt_seq, src_seq, input_mask, utt_loc):
    # HT-Encoder
    pe_utt_loc = _get_pe_inputs(tgt_seq, src_seq, input_mask, utt_loc)

    # UT-MASK
    enc_mask_utt = gen_encoder_ut_mask(src_seq, input_mask, utt_loc)

    # CT-MASK
    enc_mask_ct = ((pe_utt_loc + input_mask) != 0).unsqueeze(1).expand(-1, src_seq.shape[1], -1)  # HIER-CLS style

    # For HIER-CLS style
    dec_enc_attn_mask = ((pe_utt_loc + input_mask) != 0).unsqueeze(1).expand(-1, tgt_seq.shape[1], -1)

    return pe_utt_loc, enc_mask_utt, enc_mask_ct, dec_enc_attn_mask


def _FULL_masks(tgt_seq, src_seq, input_mask, utt_loc):
    # HT-Encoder
    pe_utt_loc = _get_pe_inputs(tgt_seq, src_seq, input_mask, utt_loc)

    # UT-MASK
    # enc_mask = input_mask.unsqueeze(1).expand(-1, src_seq.shape[1], -1)
    enc_mask_utt = gen_encoder_ut_mask(src_seq, input_mask, utt_loc)

    # CT-MASK
    enc_mask_ct = input_mask.unsqueeze(1).expand(-1, src_seq.shape[1], -1)

    dec_enc_attn_mask = input_mask.unsqueeze(1).expand(-1, tgt_seq.shape[1], -1)

    return pe_utt_loc, enc_mask_utt, enc_mask_ct, dec_enc_attn_mask


def get_hier_encoder_mask(tgt_seq, src_seq, input_mask, utt_loc, type:str):
    # Padding correction
    # No token other than padding should attend to padding
    # But padding needs to attend to padding tokens for numerical stability reasons
    utt_loc = utt_loc - 2 * input_mask * utt_loc

    # CT-Mask type
    assert type in ["hier", "cls", "full"]

    if type == "hier": # HIER: Context through final utterance
        return _HIER_masks(tgt_seq, src_seq, input_mask, utt_loc)
    elif type == "cls": # HIER-CLS: Context through cls tokens
        return _CLS_masks(tgt_seq, src_seq, input_mask, utt_loc)
    elif type == "full": # Ut-mask only, CT-mask: Full attention
        return _FULL_masks(tgt_seq, src_seq, input_mask, utt_loc)

    return None