#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:55:46 2020

@author: zlpure
"""
import numpy as np

psi_x1x2 = psi_x3x4 = np.array([[1, 0.9], [0.9, 1]])
psi_x2x3 = psi_x3x5 = np.array([[0.1, 1], [1, 0.1]])

phi_x2y2 = phi_x4y4  = np.array([[1, 0.1], [0.1, 1]])
y_2, y_4 = 0, 1

phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = np.array([0.5, 0.5])

phi_2 = phi_x2y2[:,y_2]
phi_4 = phi_x4y4[:,y_4]

message_dict = {}
m_x1x2 = m_x2x1 = m_x2x3 = m_x3x2 = m_x3x4 = m_x4x3 = m_x3x5 = m_x5x3 = np.array([0.5, 0.5])

message_dict['x1x2'] = m_x1x2
message_dict['x2x1'] = m_x2x1
message_dict['x2x3'] = m_x2x3
message_dict['x3x2'] = m_x3x2
message_dict['x3x4'] = m_x3x4
message_dict['x4x3'] = m_x4x3
message_dict['x3x5'] = m_x3x5
message_dict['x5x3'] = m_x5x3

for i in range(100):
    key = np.random.choice(list(message_dict.keys()), 1, replace=False)[0]
    #replace:True表示可以取相同数字，False表示不可以取相同数字
    if key == 'x1x2':
        message_dict[key] = np.dot(phi_1, psi_x1x2)
    elif key == 'x2x1':
        message_dict[key] = np.dot(psi_x1x2, phi_2) * message_dict['x3x2']
    elif key == 'x2x3':
        message_dict[key] = np.dot(phi_2, psi_x2x3) * message_dict['x1x2']
    elif key == 'x3x2':
        message_dict[key] = np.dot(psi_x2x3, phi_3) * message_dict['x4x3'] * message_dict['x5x3']
    elif key == 'x3x4':
        message_dict[key] = np.dot(phi_3, psi_x3x4) * message_dict['x2x3'] * message_dict['x5x3']
    elif key == 'x4x3':
        message_dict[key] = np.dot(psi_x3x4, phi_4)
    elif key == 'x3x5':
        message_dict[key] = np.dot(phi_3, psi_x3x5) * message_dict['x2x3'] * message_dict['x4x3']
    elif key == 'x5x3':
        message_dict[key] = np.dot(psi_x3x5, phi_5) 

b_1_temp = phi_1 * message_dict['x2x1']
b_1 = b_1_temp / np.sum(b_1_temp)

b_2_temp = phi_2 * message_dict['x1x2'] * message_dict['x3x2']
b_2 = b_2_temp / np.sum(b_2_temp)

b_3_temp = phi_3 * message_dict['x2x3'] * message_dict['x4x3'] * message_dict['x5x3']
b_3 = b_3_temp / np.sum(b_3_temp)
    
b_4_temp = phi_4 * message_dict['x3x4']
b_4 = b_4_temp / np.sum(b_4_temp)
    
b_5_temp = phi_5 * message_dict['x3x5']
b_5 = b_5_temp / np.sum(b_5_temp)
    
    
    