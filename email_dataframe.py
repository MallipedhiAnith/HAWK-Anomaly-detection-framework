"""
email - dataframe
"""

import pandas as pd
import numpy as np
import csv
from tabulate import tabulate
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage
import smtplib
from smtplib import SMTP
from datetime import date
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()

#my_date = date.today()
def dataframe_mail(file_list, dataframe_subj, NoAnomalyDL):
    #print("hello3")
    me = 'mallipeddi.anith@exlservice.com'
  
    you = NoAnomalyDL

    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = dataframe_subj
    msgRoot['From'] = me
    msgRoot['To'] = you

    for filename in file_list:
        with open(filename, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )

        # Add attachment to message and convert message to string
        msgRoot.attach(part)

    smtp = SMTP()
    smtp.set_debuglevel(0)
    smtp.connect('mx.lvs.exlservice.com', 25)

    smtp.sendmail(me, you.split(','), msgRoot.as_string())
    smtp.quit()