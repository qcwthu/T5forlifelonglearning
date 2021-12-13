import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
#from transformers.models.bart.modeling_bart import shift_tokens_right

from utils import label_smoothed_nll_loss

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

class MyBart(BartForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):
        #print(decoder_input_ids.shape)
        #print(self.config.pad_token_id)
        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
            #_decoder_input_ids = decoder_input_ids
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            #decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        #print(outputs[0])
        #print(outputs[0].shape)
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
            return loss
        return (lm_logits, ) + outputs[1:]

