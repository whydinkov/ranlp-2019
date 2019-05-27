import numpy as np


def _get_words(text):
    cleaned = "".join([c for c in text if c.isalnum() or c == ' '])
    return cleaned.split(" ")


def _get_sentences(text):
    sentences = text.split('.')

    return [s for s in sentences if any(s)]

# by default, don't apply filter
def _count(items, filter_func=None):
    if not filter_func:
        return len(items)

    return len([item for item in items if filter_func(item)])


def __is_spec(item):
    item = item.strip()
    return any(item) and not item.isalnum()


def __is_upper(item):
    return item.isupper()


def get_stats_text(text):
    text_words = _get_words(text)
    text_sentences = _get_sentences(text)
    avg_sentence_length = np.average([_count(s) for s in text_sentences])
    avg_sentence_length_words = np.average(
        [_count(_get_words(s)) for s in text_sentences])

    return {
        'avg_word_length_text': np.average([len(w) for w in text_words]),
        'word_count_text': _count(text_words),
        'char_count_text': _count(text),
        'spec_char_count_text': _count(text, __is_spec),
        'upper_char_count_text': _count(text, __is_upper),
        'upper_word_count_text': _count(text_words, __is_upper),
        'sentence_count_text': _count(text_sentences),
        'avg_sentence_length_char_text': avg_sentence_length,
        'avg_sentence_length_word_text': avg_sentence_length_words
    }


def get_stats_title(title):
    title_words = _get_words(title)

    return {
        'avg_word_length_title': np.average([len(w) for w in title_words]),
        'word_count_title': _count(title_words),
        'char_count_title': _count(title),
        'spec_char_count_title': _count(title, __is_spec),
        'upper_char_count_title': _count(title, __is_upper),
        'upper_word_count_title': _count(title_words, __is_upper),
    }
