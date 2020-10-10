
import requests as rq, json

def bitstamp():
    URL = 'https://www.bitstamp.net/api/ticker'
    return URL

def getData(market = bitstamp()):
    try:
        # r = rq.get(market).json()
        # return r['last']
        r = rq.get(market)
        priceFloat = float(json.loads(r.text)['last'])
        return priceFloat
    except rq.ConnectionError:
        print('Error al conectar con el mercado')

