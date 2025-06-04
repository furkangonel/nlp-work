import openai




def chat_with_gpt(prompt, history_list):
    client = openai.OpenAI(api_key="api_key") 
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sen yardımcı bir asistansın."},
            *[{"role": "user", "content": msg} for msg in history_list],
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()



if __name__ == "__main__":
    history_list = []

    while True:
        user_input = input("Kullanıcı mesajı: ")
        if user_input.lower() in ["exit", "q"]:
            print("Konuşma sona erdi.")
            break
        history_list.append(user_input)
        response = chat_with_gpt(user_input, history_list)
        print(f"Chatbot: {response}")