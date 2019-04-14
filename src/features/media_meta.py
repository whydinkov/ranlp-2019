from datetime import datetime


def _has(position):
    return position and 'липсва' not in position


def _get_popularity(media_popularity):
    if str(media_popularity).isnumeric():
        return 1 / int(media_popularity)

    return 0


def _get_days(established):
    try:
        established_date = datetime.strptime(established, '%Y-%m-%d')
        to_date = datetime(2019, 1, 1)

        return (to_date - established_date).days
    except:
        return 0


def get_stats(media):
    return {
        'editor': _has(media['editor']),
        'responsible_person': _has(media['responsible_person']),
        'bg_server': 'Bulgaria' in media['server_in'],
        'popularity': _get_popularity(media['popularity']),
        'domain_person': any(media['domain_responsible_person']),
        'days_existing': _get_days(media['established'])
    }
