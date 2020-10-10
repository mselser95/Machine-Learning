import Practica2.libs.printing as pr
import Practica2.libs.api_btc as abtc
import Practica2.libs.ticker as tk
from time import sleep

try:
    pr.clearScreen('Other')
    pr.msg("Datos desde el mercado Bitstamp:" + abtc.bitstamp())
    while True:
        tk.show(abtc.getData(abtc.bitstamp()))
        sleep(1/14)

except KeyboardInterrupt:
    pass

finally:
    pr.salida()