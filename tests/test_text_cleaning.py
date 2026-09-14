from src.data.text_cleaning import basic_tokenize, clean_tweet


def test_lowercases_text():
    assert clean_tweet("GREAT Flight") == "great flight"


def test_strips_urls():
    assert "http" not in clean_tweet("check this out https://example.com/foo")


def test_strips_mentions_but_keeps_hashtag_words():
    cleaned = clean_tweet("@united this is #unacceptable")
    assert "united" not in cleaned
    assert "unacceptable" in cleaned


def test_collapses_elongated_characters():
    cleaned = clean_tweet("soooo goooood")
    assert "sooo" not in cleaned  # collapsed to at most 2 repeats
    assert "soo" in cleaned


def test_removes_punctuation():
    cleaned = clean_tweet("Wow!!! Really?? #@$%")
    assert "!" not in cleaned and "?" not in cleaned


def test_handles_non_string_input_gracefully():
    assert clean_tweet(None) == ""
    assert clean_tweet(float("nan")) == ""


def test_basic_tokenize_splits_on_whitespace():
    tokens = basic_tokenize("This flight was Great!")
    assert tokens == ["this", "flight", "was", "great"]


def test_empty_string_produces_no_tokens():
    assert basic_tokenize("   ") == []
