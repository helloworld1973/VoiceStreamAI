from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class TransformersMBartLTM():
    def __init__(self):
        # load the model and embedding encoder
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    async def translate(self, complete_txt, lang):
        # Check the language and set translation direction
        if lang == "zh":
            src_lang = "zh_CN"
            target_lang = "en_XX"
        elif lang == "en":
            src_lang = "en_XX"
            target_lang = "zh_CN"
        else:
            raise ValueError("Unsupported language for translation.")

        # Set the source language for the tokenizer
        self.tokenizer.src_lang = src_lang
        # Encode the text
        encoded_text = self.tokenizer(complete_txt, return_tensors="pt")

        # Generate translation tokens with the target language
        generated_tokens = self.model.generate(
            **encoded_text,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang]
        )

        # Decode the generated tokens to get the translated text
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return translated_text



