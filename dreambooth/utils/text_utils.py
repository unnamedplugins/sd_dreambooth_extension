import re
from typing import List

import torch
import torch.utils.checkpoint
from transformers import CLIPTextModel

def smart_split_input_ids(input_ids, pad_tokens, b_size, max_token_length,
                          tokenizer_max_length):
    comma_token = 267
    start_token = input_ids[0]
    end_token = input_ids[-1]
    size_limit = tokenizer_max_length - 2
    batch_id_list : List[List[int]] = input_ids.tolist()
    for prompt in batch_id_list:
        prompt_parts = torch.tensor([], device=input_ids.device)
        chunks = max_token_length / size_limit
        for x in range(0, chunks):
            start = x * size_limit + 1
            if start > len(prompt):
                padding = torch.full(tokenizer_max_length, end_token, device=input_ids.device)
                padding[0] = start_token
                prompt_parts = torch.cat((prompt_parts, ))

            chunk = prompt[start:(start + size_limit)]
            if len(chunk) > size_limit:
                if comma_token in chunk:
                    last_idx = chunk[::-1].index(comma_token)

            start = 0
            found = prompt.index(comma_token, start, start + size_limit)


# Implementation from https://github.com/bmaltais/kohya_ss
def encode_hidden_state(text_encoder: CLIPTextModel, input_ids, pad_tokens, b_size, max_token_length,
                        tokenizer_max_length, clip_skip):

    # smart_split_input_ids(input_ids,
    #                       pad_tokens,
    #                       b_size,
    #                       max_token_length,
    #                       tokenizer_max_length)

    # extend = (self.max_tokens + 2) - input_ids.shape[-1]
    # if extend > 0:
    #     input_ids = torch.cat((input_ids, torch.full((1,extend), input_ids[0][-1])), dim=1)
    # elif extend < 0:
    #     endtoken = input_ids[0, -1]
    #     input_ids = input_ids[:,0:(self.max_tokens+2)]
    #     input_ids[0, -1] = endtoken

    # if input_ids.shape[-1] > max_token_length:



    if pad_tokens:
        input_ids = input_ids.reshape((-1, tokenizer_max_length))  # batch_size*3, 77

    if clip_skip <= 1:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out['hidden_states'][-clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    if not pad_tokens:
        return encoder_hidden_states

    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if max_token_length > 75:
        sts_list = [encoder_hidden_states[:, 0].unsqueeze(1)]
        for i in range(1, max_token_length, tokenizer_max_length):
            sts_list.append(encoder_hidden_states[:, i:i + tokenizer_max_length - 2])
        sts_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
        encoder_hidden_states = torch.cat(sts_list, dim=1)

    return encoder_hidden_states


def prompt_to_tags(src_prompt: str, instance_token: str = None, class_token: str = None) -> List[str]:
    src_tags = src_prompt.split(',')
    if class_token:
        conjunctions = ['a ', 'an ', 'the ']
        src_tags = [tag.replace(conjunction + class_token, '') for tag in src_tags for conjunction in conjunctions]
    if class_token and instance_token:
        src_tags = [tag.replace(instance_token, '').replace(class_token, '') for tag in src_tags]
    src_tags = [' '.join(tag.split()) for tag in src_tags]
    src_tags = [tag.strip() for tag in src_tags if tag]
    return src_tags


def build_strict_tokens(
        caption: str = '',
        tenc_start_token: str = '',
        tenc_end_token: str = ''
):
    caption_list = []
    caption_split = re.split(r'[,;.!?]\s', caption)

    for cap in caption_split:
        words_with_special_token = []
        split_cap = cap.split(" ")

        for sc in split_cap:
            if sc: words_with_special_token.append(f"{sc}</w>")

        new_cap = ' '.join(words_with_special_token)
        caption_list.append(f"{tenc_start_token}{new_cap}{tenc_end_token}")

    special_caption = ', '.join(caption_list)

    return special_caption
