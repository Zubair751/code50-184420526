import sys

from PIL import Image, ImageDraw, ImageFont

MODEL = "bert-base-uncased"
K = 3
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def get_mask_token_index(mask_token_id, inputs):
    try:
        ids = inputs["input_ids"].numpy()[0]
    except Exception:
        ids = inputs["input_ids"][0].tolist()
    for i, token_id in enumerate(ids):
        if int(token_id) == int(mask_token_id):
            return i
    return None


def get_color_for_attention_score(attention_score):
    value = int(float(attention_score) * 255)
    return (value, value, value)


def visualize_attentions(tokens, attentions):
    for layer_num, layer in enumerate(attentions):
        for head_num in range(layer.shape[1]):
            generate_diagram(
                layer_num + 1,
                head_num + 1,
                tokens,
                layer[0][head_num]
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)
    for i, token in enumerate(tokens):
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token, fill="white", font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token, fill="white", font=FONT
        )
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


def main():
    text = input("Text: ")

    try:
        import torch
        from transformers import AutoTokenizer, BertForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        inputs = tokenizer(text, return_tensors="pt")
        mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
        if mask_token_index is None:
            sys.exit(f"Input must include mask token {tokenizer.mask_token}.")
        model = BertForMaskedLM.from_pretrained(MODEL)
        with torch.no_grad():
            result = model(**inputs, output_attentions=True)
        mask_token_logits = result.logits[0, mask_token_index]
        top_tokens = torch.topk(mask_token_logits, K).indices.tolist()
        for token in top_tokens:
            print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
        visualize_attentions(inputs.tokens(), result.attentions)
    except ImportError:
        import tensorflow as tf
        from transformers import AutoTokenizer, TFBertForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        inputs = tokenizer(text, return_tensors="tf")
        mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
        if mask_token_index is None:
            sys.exit(f"Input must include mask token {tokenizer.mask_token}.")
        model = TFBertForMaskedLM.from_pretrained(MODEL)
        result = model(**inputs, output_attentions=True)
        mask_token_logits = result.logits[0, mask_token_index]
        top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
        for token in top_tokens:
            print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
        visualize_attentions(inputs.tokens(), result.attentions)


if __name__ == "__main__":
    main()
