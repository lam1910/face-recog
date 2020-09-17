# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:30:07 2020

@author: lam Nguyen Ngoc
"""
from unidecode import unidecode


# convert from tiếng việt có dấu to tieng viet khong dau
def convert(s):
    return unidecode(s)
