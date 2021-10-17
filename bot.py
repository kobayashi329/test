import requests
def post():
    files = {'file':open(image.jpg,'rb')}
    params = {
            'token':TOKEN, 
            'channels':CHANNEL_TOKEN,
            'filename':"filename",
            'initial_comment': "画像についてのコメント",
            'title': "ファイルの名前"
    }
    requests.post(url="https://slack.com/api/files.upload",params=params,