import time

import openai


class ChatGPT:
    def __init__(self, api_base, api_key, model, prompts, conversation_track=False, stream=False):
        openai.api_key = api_key
        openai.base_url = api_base
        openai.default_headers = {"x-foo": "true"}
        self.conversation_track = conversation_track
        self.conversations = {}
        self.prompts = prompts
        self.model = model
        self.stream = stream

    def __call__(self, text_message, user_id=None):
        """
        Make remember all the conversation
        :param old_model: Open AI model
        :param user_id: telegram user id
        :param text_message: text message
        :return: str
        """
        if not self.conversation_track:
            # Generate response
            return self.generate_response_chatgpt([{"role": "user", "content": text_message}])

        conversation_history, gpt_responses = [], []
        # Get the last 10 conversations and responses for this user
        user_conversations = self.conversations.get(user_id, {'conversations': [], 'responses': []})
        user_messages = user_conversations['conversations'] + [text_message]
        gpt_responses = user_conversations['responses']

        # Store the updated conversations and responses for this user
        self.conversations[user_id] = {'conversations': user_messages, 'responses': gpt_responses}

        # Construct the full conversation history in the user:assistant, " format
        for i in range(min(len(user_messages), len(gpt_responses))):
            conversation_history.append({
                "role": "user", "content": user_messages[i]
            })
            conversation_history.append({
                "role": "assistant", "content": gpt_responses[i]
            })

        # Add last prompt
        conversation_history.append({
            "role": "user", "content": text_message
        })

        # Generate response
        response = self.generate_response_chatgpt(conversation_history)

        # Add the response to the user's responses
        gpt_responses.append(response)
        # Store the updated conversations and responses for this user
        self.conversations[user_id] = {'conversations': user_messages, 'responses': gpt_responses}
        return response

    def generate_response_chatgpt(self, message_list):
        if self.stream:
            try:
                stream = openai.chat.completions.create(
                    model=self.model,
                    messages=self.prompts + message_list,
                    stream=True
                )
            except Exception as e:
                time.sleep(1)
                print(e)
                return self.generate_response_chatgpt(message_list)
            return stream
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=self.prompts + message_list
            )
        except Exception as e:
            time.sleep(1)
            print(e)
            return self.generate_response_chatgpt(message_list)
        if not hasattr(response, 'choices'):
            time.sleep(1)
            return self.generate_response_chatgpt(message_list)
        return response.choices[0].message.content.strip()
