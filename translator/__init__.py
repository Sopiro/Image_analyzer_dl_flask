import requests


def kakao_translator(sentence, api_key):
    headers = {'Authorization': 'KakaoAK {}'.format(api_key)}
    params = {'query': sentence, 'src_lang': 'en', 'target_lang': 'kr'}
    api_response = requests.post(url='https://dapi.kakao.com/v2/translation/translate', headers=headers, data=params)

    return api_response.json()['translated_text'][0][0]
