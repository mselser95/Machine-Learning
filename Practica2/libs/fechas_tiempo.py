import datetime as dtime

def getTiempo():
    return dtime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]