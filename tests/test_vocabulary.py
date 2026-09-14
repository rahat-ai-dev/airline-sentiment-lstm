import json

from src.features.vocabulary import Vocabulary, pad_sequences


def test_fit_builds_pad_and_oov_tokens():
    vocab = Vocabulary(max_vocab_size=10, max_sequence_length=5)
    vocab.fit(["good flight", "bad flight", "good crew"])
    assert vocab.token_to_id[vocab.pad_token] == 0
    assert vocab.token_to_id[vocab.oov_token] == 1


def test_encode_maps_unknown_words_to_oov():
    vocab = Vocabulary(max_vocab_size=10, max_sequence_length=5)
    vocab.fit(["good flight"])
    encoded = vocab.encode("terrible flight")
    assert encoded[1] == vocab.token_to_id["flight"]
    assert encoded[0] == vocab.oov_id  # "terrible" never seen during fit


def test_encode_batch_produces_fixed_width_matrix():
    vocab = Vocabulary(max_vocab_size=50, max_sequence_length=4)
    vocab.fit(["a b c d e", "x y"])
    matrix = vocab.encode_batch(["a b c d e f", "x"])
    assert matrix.shape == (2, 4)


def test_pad_sequences_truncates_long_and_pads_short():
    padded = pad_sequences([[1, 2, 3, 4, 5], [1]], max_len=3, pad_value=0)
    assert padded.tolist() == [[1, 2, 3], [1, 0, 0]]


def test_save_and_load_round_trip(tmp_path):
    vocab = Vocabulary(max_vocab_size=20, max_sequence_length=6)
    vocab.fit(["great service", "terrible delay"])

    path = tmp_path / "vocab.json"
    vocab.save(path)
    assert json.loads(path.read_text())["token_to_id"] == vocab.token_to_id

    loaded = Vocabulary.load(path)
    assert loaded.token_to_id == vocab.token_to_id
    assert loaded.max_sequence_length == vocab.max_sequence_length
    assert loaded.encode("great service") == vocab.encode("great service")
