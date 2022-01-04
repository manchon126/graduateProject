#전체 계좌 조회

!pip uninstall JWT
!pip uninstall PyJWT
!pip install PyJWT

import os
import jwt
import uuid
import hashlib
from urllib.parse import urlencode

import requests

os.environ['UPBIT_OPEN_API_ACCESS_KEY'] = ""
access_key = os.environ['UPBIT_OPEN_API_ACCESS_KEY']
os.environ['UPBIT_OPEN_API_SECRET_KEY'] = ""
secret_key = os.environ['UPBIT_OPEN_API_SECRET_KEY']
os.environ['UPBIT_OPEN_API_SERVER_URL'] = "https://api.upbit.com"
server_url = os.environ['UPBIT_OPEN_API_SERVER_URL']

payload = {
    'access_key': access_key,
    'nonce': str(uuid.uuid4()),
}

jwt_token = jwt.encode(payload, secret_key)
authorize_token = 'Bearer {}'.format(jwt_token)
headers = {"Authorization": authorize_token}

res = requests.get(server_url + "/v1/accounts", headers=headers)

print(res.json())
