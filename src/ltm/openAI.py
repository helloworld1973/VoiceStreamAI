from openai import OpenAI


class openAI_QA():

    def translate(self, complete_txt):
        client = OpenAI(api_key='')

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who translates from Maori to English."},
                {"role": "user",
                 "content": f"Please translate the Maori language '{complete_txt}' to English. No need to explain it, just send me with the format XXX"}
            ]
        )

        translated_text = response.choices[0].message.content
        return translated_text
