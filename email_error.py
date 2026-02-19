"""
email - error
"""

import pandas as pd
import numpy as np
import csv
from tabulate import tabulate
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
from smtplib import SMTP
from datetime import date
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()

#my_date = date.today()
def error(text,subject):
    #print("hello3")
    me = 'mallipeddi.anith@exlservice.com'
    #server = 'atom.paypalcorp.com'
    you = 'mallipeddi.anith@exlservice.com'
    #you = 'dl-pp-risk-credit-fraud-analytics@paypal.com,01418c83.paypal.onmicrosoft.com@amer.teams.ms'

    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = subject
    msgRoot['From'] = me
    msgRoot['To'] = you

    part1 = MIMEText(text, "plain")
    msgRoot.attach(part1)

    smtp = SMTP()
    smtp.set_debuglevel(0)
    smtp.connect('mx.lvs.exlservice.com', 25)

    smtp.sendmail(me, you.split(','), msgRoot.as_string())
    smtp.quit()