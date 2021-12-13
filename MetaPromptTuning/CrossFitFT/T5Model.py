import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
#from transformers.models.bart.modeling_bart import shift_tokens_right

from utils import label_smoothed_nll_loss

class T5FTModel(nn.Module):

    def __init__(self, args, inermodel):
        super(T5FTModel, self).__init__()
        self.args = args
        self.model = inermodel
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id

    def forward(self, input_ids, attention_mask=None,
                decoder_input_ids=None, decoder_attention_mask=None, is_training=False):
        ###tokenizer.pad_token_id 0
        lm_labels = decoder_input_ids
        lm_labels[lm_labels[:, :] == 0] = -100

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask
        )

        if is_training:
            loss = outputs[0]
            return loss
        return outputs[0]

    def _generative_step(self, input_ids, attention_mask, num_beams, max_length):
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=input_embed_part,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            #decoder_attention_mask=batch['target_mask'],
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        return generated_ids

