import datetime

def time_stamp():
    now = datetime.datetime.now()
    month =  now.strftime("%B").upper()[0:3]
    year  = now.strftime("%Y")[-2:]
    day = now.strftime("%d")
    hours = now.strftime("%H%M")+"C"
    timestamp = day + hours + month + year
    return timestamp