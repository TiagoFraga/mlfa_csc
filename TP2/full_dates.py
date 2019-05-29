#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:25:19 2019

@author: tiagofraga
"""

import csv
import datetime

hour = 1

years = [2018,2019]
months = [10,11,12,1,2,7,8,9]
days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
hours = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

with open('full_dates.csv', mode='w') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['match_hour'])
    
    for year in years:
        for month in months:
            for day in days:
                for hour in hours:
                    full = str(year) + "-" + str(month) + "-" + str(day) + "-" + str(hour)
                    writer.writerow([full])
                    
            

    