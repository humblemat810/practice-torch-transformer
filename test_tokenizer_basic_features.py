from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize the tokenizer
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

# Use whitespace pre-tokenizer
tokenizer.pre_tokenizer = Whitespace()

# Create a trainer for the WordLevel model
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"])

# Example training data
corpus = [
    "Hello, how are you?",
    "I am a student.",
    "This is an example sentence.",
    "We are learning how to train a tokenizer."
]

# Train the tokenizer on the corpus
tokenizer.train_from_iterator(corpus, trainer)
tokenizer.id_to_token(0)
# Add the [PAD] token if not present
tokenizer.model.add_tokens(["[PAD]"])

# Get the token ID for [PAD]
pad_token_id = tokenizer.token_to_id("[PAD]")

print(f"Token ID for [PAD]: {pad_token_id}")

# Verify by converting the ID back to the token
token_from_id = tokenizer.id_to_token(pad_token_id)

print(f"Token from ID {pad_token_id}: {token_from_id}")
