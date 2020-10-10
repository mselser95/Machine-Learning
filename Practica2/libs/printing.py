# -*- coding: utf-8 -*-


import os
def clearScreen(SO='Win'):
    if SO == 'Win':
        os.system('cls')
    elif SO == 'Other':
        os.system('clear')

def salida():
    print("Hasta la vista baby")

def saltoLinea():
    print('')

def msg(text = 'MLPY...'):
    print('\n' + str(text) + '\n')

