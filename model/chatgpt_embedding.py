import openai

openai.api_key = "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"
openai_embedding_model = "text-embedding-ada-002"

def get_embedding(text: str, model: str):
    """
    Get embedding from text using model
    :param text: text to embed
    :param model: model to use
    :return: embedding
    """
    result = openai.Embedding.create(
        model = model,
        input = text
    )
    return result["data"][0]["embedding"]

if __name__ == '__main__':
    print(get_embedding("Hello world", openai_embedding_model))