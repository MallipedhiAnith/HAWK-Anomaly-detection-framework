"""
email - charts
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
from PIL import Image
import random
import urllib.request
import io
import os

def send_email(dictTemp, mailpoints, country, severity, emailInsight, metrictype, product1, cc, dashboardlink, extra, highRiskDL, LowRiskDL, NoAnomalyDL, isDollar, pi_flag):
    me = 'Mallipeddi.anith@exlservice.com'
    if((severity == 'HIGHLY CRITICAL') or (severity == 'CRITICAL') or (severity == 'HIGH')):
        you = highRiskDL
    elif((severity == 'LOW') or (severity == 'MEDIUM')):
        you = LowRiskDL
    else:
        you = NoAnomalyDL
    
    cutmail = you[: you.index('@') ]
    if(isDollar == 'Y'):
        subject = "[HAWK] "+severity+" - "+country+' '+product1+' '+metrictype+' supress view ('+extra+')'+" on "+str(cc)
    else:
        subject = "[HAWK] "+severity+" - "+country+' '+product1+' '+metrictype+' '+extra+" on "+str(cc)
        
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = subject
    msgRoot['From'] = me
    msgRoot['To'] = you
    txcol = 'white'
    header_v2 = 'Header.png'
    line_v2 = 'Line.png'
    line2_v2 = 'Line2.png'
    footer_v2 = 'Footer.png'

    if(severity == 'HIGHLY CRITICAL'):
        bgcol0 = '#DC0064'
        bgcol1 = '#640587'
    elif(severity == 'CRITICAL'):
        bgcol0 = '#FF9700'
        bgcol1 = '#EA4672'
    elif(severity == 'HIGH'):
        bgcol0 = '#3CD3A0'
        bgcol1 = '#32A6E2'
    elif(severity == 'MEDIUM'):
        bgcol0 = '#3CD3A0'
        bgcol1 = '#32A6E2'
    elif(severity == 'LOW'):
        bgcol0 = '#898989'
        bgcol1 = '#474747'
    else:
        bgcol0 = '#898989'
        bgcol1 = '#474747'

    htmlMain = """<html>head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"><style>
    html,body {
        height:100%;
        width:410mm;
        align-content: center;
        text-align: center;
        justify-content: center;
        font-family: Arial;
    }

    #myheader{
        background-color: DimGray;
        color: white;
        font-family: Arial;
    }

    p { margin:1 }

    table.dataframe{
        table-layout: fixed;
        background-color:white;
        width:100%;
        font-size:15px;
        font-family: Arial;
    }

    table{
        table-layout: fixed;
        border-collapse: collapse;
        background-color:white;
        align:center;
        width:100%;
        font-size:13px;
        font-family: Arial;
    }

    table.normal{
        width:100%;
        background-color:white;
        font-size:15px;
        font-family: Arial;
    }
    """
    
    color_format = '{' + f"""background-color:{bgcol1} ; font-size:15px ; font-family: Arial;""" + '}'
    htmlMain += f"""th.insight{color_format}"""
    htmlMain +="""
    td.insights {
        border: 4px solid #441782;
        border-style:single;
        font-size:13px;
        font-family: Arial;
    }
    td.insi{
        text-align:center;
        font-family: Arial;
    }
    td {
        border: 1px solid black;
        text-align: center;
        font-size:13px;
        font-family: Arial;
        overflow-wrap: break-word;
        display: fixed;
        justify-content: center;
        padding:3px 10px 3px 3px;
    }
    #statsTable{
        table-layout: fixed;
        width:100%
    }
    #statsTable td{
        border: 1px solid black;
        text-align: center;
        font-size:13px;
        font-family: Arial;
    }
    tr.noBorder td {
        border: 0;
    }
    td.dataframe{
        table-layout: fixed;
        width:100%
        text-align: center;
        border: 1px solid DimGray;
        font-size:13px;
        font-family: Arial;
        padding:0px 10px 0px 2px;
    }
    """
    
    color_format = '{' + f"""border: 1px solid black;
        text-align: center;
        background-color: {bgcol1};
        color: white;
        font-size:14px;
        font-family: Arial;
        padding:3px 10px 3px 3px;""" + '}'
    
    htmlMain += f"""th {color_format}"""
    htmlMain += """
    h2 {
        margin-bottom: 0px;
        font-family: Arial;
    }
    </style></head><body>
    <table cellpadding="0" border="0" cellspacing="0" style="padding:0px;margin:0px;">
    <tr class = 'noBorder'><td colspan="3" style="padding:0px;margin:0px;font-size:20px;height:20px;background-color:white; height="20">&nbsp;</td></tr>
    <tr>
    <td style="padding:0px;margin:0px;text-align:center;border: 0;" width="1200">
    """
    
    i=0
    myimagelist=[]
    msgText1=""
    newing = header_v2
    fp = open(newing, 'rb')
    msgImages=MIMEImage(fp.read(),_subtype="jpeg")
    fp.close()
    msgText1+= '<img src="cid:image{}" alt=""  width="100%" />'.format(i)
    myimagelist.append(msgImages)
    i+=1
    html = """{htmlimagecode}"""
    htmlMain+=html.format(htmlimagecode = msgText1)
    
    lineing = ''
    newing1 = line_v2
    fp = open(newing1, 'rb')
    msgImages=MIMEImage(fp.read())
    fp.close()
    lineing+= '<img src="cid:image{}" width=100% /><br>'.format(i)
    myimagelist.append(msgImages)
    i+=1
    
    lineing2 = ''
    newing2 = line2_v2
    fp = open(newing2, 'rb')
    msgImages=MIMEImage(fp.read())
    fp.close()
    lineing2+= '<img src="cid:image{}" width=100% /><br>'.format(i)
    myimagelist.append(msgImages)
    i+=1
    
    html="""<table border =.5 style="text-align:center ; border-collapse: collapse ; width:100% ;" class='normal' >
            <tr border =.5 >
            <td rowspan =2 bgcolor={bgcol} class ='insi' style="font-size:1.5em; color:{txcol};text-align:center;" ><span style="font-weight:bold" ><font face="Arial">{severity}</font></span></td>
            <th class='insight'>REGION</th>
            <th class='insight'>PRODUCT</th>
            <th class='insight'>METRIC-CHECKPOINT</th>
            <th class='insight'>PERIOD</th>
            </tr>
            <tr>
            <td class ='insi' style="text-align:center;"><b>{country}</b></td>
            <td class ='insi' style="text-align:center;"><b>{PPC}</b></td>
            <td class ='insi' style="text-align:center;"><b>{Acquisition}</b></td>
            <td class ='insi' style="text-align:center;"><b>{cc}</b></td>
            </tr>
            </table>"""
            
    htmlMain+=html.format(bgcol = bgcol0,txcol=txcol,severity=severity,country=country,PPC=product1,Acquisition=metrictype,cc=cc)
    htmlMain+="""<br><table width=100%><tr border = 1> <td class ='insights' style="text-align:left;"> <h2 style="color:{bgcol} ; margin:0.7em;">&nbsp&nbspKEY INSIGHTS </h2>{overall_insight}</td></tr></table><br>"""
    htmlMain = htmlMain.format(bgcol=bgcol1, overall_insight = mailpoints)
    htmlMain += emailInsight.to_html(index=True, col_space=1000, escape=False,table_id="statsTable")
    html += """<br><p style="text-align:left;font-size:12px;">Note:-</p>
            <p style="text-align:left;font-size:12px;">&nbsp&nbsp&nbsp&nbsp&nbsp&nbspFor favorable business metrics like approval rate upward delta represented in '<font color='green'>&#9650;</font>' and downward represented in '<font color='red'>&#9660;</font>'.</p>
            <p style="text-align:left;font-size:12px;">&nbsp&nbsp&nbsp&nbsp&nbsp&nbspFor unfavorable business metrics like fraud decline rate upward delta represented in '<font color='red'>&#9650;</font>' and downward represented in '<font color='green'>&#9660;</font>'.</p>"""
    htmlMain += html.format()
    
    print('entered email')
    for list_ele in dictTemp:
        for ele in list_ele[1]:
            data_ele=ele[1]
            filename_ele=ele[3]
            title_ele=ele[0]
            tabletitle=ele[2]
            styles="""
            table{
                border: 1px solid DimGray;
                text-align: center;
                width:100%
            }
            th, td {
                text-align: center;
                border: 1px solid DimGray;
            }
            tr:hover {background-color: LightGray;}
            """
            color_format = '{' + f"""background-color:{bgcol1} ; color: white;""" + '}'
            styles += f"""th {color_format} """
            msgText1=""
            for k in range(len(filename_ele)):
                fp = open(filename_ele[k], 'rb')
                msgImages=MIMEImage(fp.read())
                fp.close()
                msgText1+= '<img src="cid:image{}" width=100% />'.format(i)
                myimagelist.append(msgImages)
                i+=1
            
            pattern = ele[4]
            if isinstance(data_ele, pd.DataFrame) and (data_ele.empty != True):
                if (pi_flag==1) and (pattern !=''):
                    print('With DF-------')
                    html = """
                    <br>
                    {lineing}
                    <div border=0><h2 align="center"><b>{var1}</b></h2></div>
                    {htmlimagecode}
                    <h2 style="color:{bgcol}; margin-bottom: 2px; text-align: center;">&nbspRECOGNIZED PATTERNS </h2>
                    {pi}
                    <br>
                    
                    <p align = 'center'>{tabletitle}</p>
                    <p align = 'center'>
                    """
                    htmlMain += html.format(lineing=lineing2, tabletitle=tabletitle, var1=title_ele, htmlimagecode=msgText1, bgcol=bgcol1, pi=pattern)
                    htmlMain += tabulate(data_ele, headers="keys", tablefmt="html", showindex=False)
                    htmlMain += "</p>"
                    htmlMain += """<table width=100% style="border: 2px solid #1e477a; border-spacing: 12px;"><tr><td class = 'patterns' style="text-align:left; padding: 12px;"><h2 style="color:{bgcol}; margin-bottom: 2px; text-align: center;">&nbspRECOGNIZED PATTERNS </h2>
                                <font face ="Arial">
                                {pi}</font></td></tr></table>"""
                else:
                    html = """
                    <br>
                    {lineing}
                    <div border=0><h2 align="center"><b>{var1}</b></h2></div>
                    {htmlimagecode}
                    <br>
                    <p align = 'center'>{tabletitle}</p>
                    <p align = 'center'>
                    """
                    htmlMain += htmlMain + html.format(lineing=lineing2, tabletitle=tabletitle, var1=title_ele, htmlimagecode=msgText1)
                    htmlMain += tabulate(data_ele, headers="keys", tablefmt="html", showindex=False)
                    htmlMain += "</p>"
            elif (pi_flag==1) and (pattern !=''):
                print("Without DF-------")
                html = """
                <br>
                {lineing}
                <div border=0><h2 align="center"><b>{var1}</b></h2></div>
                {htmlimagecode}
                <br>
                <h2 style="color:{bgcol}; margin-bottom: 2px; text-align: center;">&nbspRECOGNIZED PATTERNS </h2>
                {pi}
                <br>
                """
                htmlMain += html.format(lineing=lineing2, var1=title_ele, htmlimagecode=msgText1, bgcol=bgcol1, pi=pattern)
            else:
                print('Without DF & PATTERN-------')
                html = """
                <br>
                {lineing}
                <div border=0><h2 align="center"><b>{var1}</b></h2></div>
                <br>
                {htmlimagecode}
                <br>
                """
                htmlMain += html.format(lineing=lineing2, var1=title_ele, htmlimagecode=msgText1)

    html = """<br>{lineing}"""
    htmlMain += html.format(lineing=lineing)
    footer1=""
    msgText1=''
    newing = footer_v2
    fp = open(newing, 'rb')
    msgImages=MIMEImage(fp.read())
    fp.close()
    msgText1+= '<img src="cid:image{}" width=100% /><br>'.format(i)
    myimagelist.append(msgImages)
    i+=1
    
    #if(dashboardlink!=''):
        #html = """
        #<br><p align = 'center' style="font-size:18px;"><b>LINK FOR DASHBOARD</b>&nbsp&nbsp&nbsp<a href={dashboardlink}><u>{dashboardlink}</u></a>
        #<a>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspALERT - <a href='https://dlmanager.paypalcorp.com/dl/{inputDL}/subscribe'>SUBSCRIBE ME </a><br>
        #"""
        #htmlMain += html.format(country = country, dashboardlink=dashboardlink, inputDL=cutmail)
    #else:
        #html = """<br><p align = 'center' style="font-size:18px;"><b>ALERT - <a href='https://dlmanager.paypalcorp.com/dl/{inputDL}/subscribe'>SUBSCRIBE ME </a></b></p><br>"""
        #htmlMain += html.format(inputDL = cutmail)
        
    html = """
    <a href = 'go/hawk/info'>{htmlimagecode}</a>
    </td>
    </tr>
    </table>
    </body>
    </html>"""
    
    htmlMain += html.format( htmlimagecode = msgText1)
    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)
    htmlMain = MIMEText(htmlMain, 'html')
    msgAlternative.attach(htmlMain)
    
    for j in range(len(myimagelist)):
        myimagelist[j].add_header('Content-ID', '<image{}>'.format(j))
        msgRoot.attach(myimagelist[j])
        
    smtp = SMTP()
    smtp.set_debuglevel(0)
    smtp.connect('mx-int.g.exlservice.com', 25)
    smtp.sendmail(me, you.split(','), msgRoot.as_string())
    smtp.quit()
    
    os.remove("Header.png")
    os.remove("Line.png")
    os.remove("Line2.png")
    os.remove("Footer.png")