from src.features import text_meta, media_meta
from nose.tools import eq_, assert_almost_equal as aeq_
from nose.tools import assert_true, assert_false


def test_text_meta():
    article = {
        'title': 'EXAMPLE TITLE text!!!',
        'text': ('This is just a small example. Of what can an article be...'
                 "That's why we try to make it AS SIMPLE AS POSSIBLE. Some "
                 'numeric test - 42.')
    }

    result = text_meta.get_stats(article)

    aeq_(result['avg_word_length_title'], 5.33, places=2)
    aeq_(result['avg_word_length_text'], 3.666, places=2)
    eq_(result['word_count_title'], 3)
    eq_(result['word_count_text'], 27)
    eq_(result['char_count_title'], 21)
    eq_(result['char_count_text'], 133)
    eq_(result['spec_char_count_title'], 3)
    eq_(result['spec_char_count_text'], 8)
    eq_(result['upper_char_count_title'], 12)
    eq_(result['upper_word_count_title'], 2)
    eq_(result['upper_char_count_text'], 22)
    eq_(result['upper_word_count_text'], 4)
    eq_(result['sentence_count_text'], 4)
    aeq_(result['avg_sentence_length_char_text'], 31.75, places=2)
    aeq_(result['avg_sentence_length_word_text'], 7.5, places=2)


def test_media_meta():
    media = {
        'editor': 'липсва на сайта',
        'server_in': 'Bulgaria',
        'established': '2015-01-01',
        'domain_responsible_person': 'Existing person',
        'popularity': 10,
        'responsible_person': 'Existing person'
    }

    result = media_meta.get_stats(media)

    assert_false(result['editor'])
    assert_true(result['responsible_person'])
    assert_true(result['bg_server'])
    aeq_(result['popularity'], 0.1, places=1)
    eq_(result['days_existing'], 1461)
