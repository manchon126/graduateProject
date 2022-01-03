#매수 주문
#주문량, 주문 가격 지정

#!pip uninstall JWT
#!pip uninstall PyJWT
!pip install PyJWT

import os
import jwt
import uuid
import hashlib
from urllib.parse import urlencode

import requests

os.environ['UPBIT_OPEN_API_ACCESS_KEY'] = "UC4uPBtI1nIGGS8ZYIXjPagQBiOAxKx4LJMkWgx9"
access_key = os.environ['UPBIT_OPEN_API_ACCESS_KEY']
os.environ['UPBIT_OPEN_API_SECRET_KEY'] = "wi1dRxWpYXGTmYQBOjuZr5SOHukHSxL2TZFYU2Rr"
secret_key = os.environ['UPBIT_OPEN_API_SECRET_KEY']
os.environ['UPBIT_OPEN_API_SERVER_URL'] = "https://api.upbit.com"
server_url = os.environ['UPBIT_OPEN_API_SERVER_URL']

query = {
    'market': 'KRW-BTC',
    'side': 'bid',
    'volume': '0.00008172',
    'price': '61800000.0',
    'ord_type': 'limit',
}
query_string = urlencode(query).encode()

m = hashlib.sha512()
m.update(query_string)
query_hash = m.hexdigest()

payload = {
    'access_key': access_key,
    'nonce': str(uuid.uuid4()),
    'query_hash': query_hash,
    'query_hash_alg': 'SHA512',
}

jwt_token = jwt.encode(payload, secret_key)
authorize_token = 'Bearer {}'.format(jwt_token)
headers = {"Authorization": authorize_token}

res = requests.post(server_url + "/v1/orders", params=query, headers=headers)

print(res.json())