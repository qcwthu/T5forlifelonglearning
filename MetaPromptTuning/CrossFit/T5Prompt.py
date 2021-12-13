import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
#from transformers.models.bart.modeling_bart import shift_tokens_right

from utils import label_smoothed_nll_loss

class T5PromptModel(nn.Module):

    def __init__(self, args, inermodel):
        super(T5PromptModel, self).__init__()
        self.args = args
        self.model = inermodel
        t5ckpt = torch.load(args.lm_adapted_path)
        self.model.load_state_dict(t5ckpt)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.promptnumber = 0
        self.promptembedding = None

    def set_prompt_embedding(self,promptnumber,promptembedding):
        self.promptnumber = promptnumber
        self.promptembedding = nn.parameter.Parameter(promptembedding)

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((attention_mask.shape[0], self.promptnumber), 1).to(self.args.device)
        all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)
        return self.model(
            inputs_embeds=allembedding,
            attention_mask=all_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def forward(self, input_ids, attention_mask=None,
                decoder_input_ids=None, decoder_attention_mask=None, is_training=False):
        ###tokenizer.pad_token_id 0
        lm_labels = decoder_input_ids
        lm_labels[lm_labels[:, :] == 0] = -100

        outputs = self._step(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            decoder_attention_mask=decoder_attention_mask
        )

        if is_training:
            loss = outputs[0]
            return loss
        return outputs[0]

    def _generative_step(self, input_ids, attention_mask, num_beams, max_length):

        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((attention_mask.shape[0], self.promptnumber), 1).to(self.args.device)
        all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)
        decoder_input_ids = (
                torch.ones((input_ids.shape[0], 1), dtype=torch.long,
                           device=input_ids.device) * self.decoder_start_token_id_use
        )

        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            #decoder_attention_mask=batch['target_mask'],
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        return generated_ids

