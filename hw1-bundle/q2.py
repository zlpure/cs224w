#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:34:38 2020

@author: zlpure
"""
import snap
import numpy as np
import matplotlib.pyplot as plt


def calcFeaturesNode(Node, Graph):
    """
    return: 1. the degree of v, i.e., deg(v);
            2. the number of edges in the egonet of v, where egonet of v is defined as the subgraph of G
            induced by v and its neighborhood;
            3. the number of edges that connect the egonet of v and the rest of the graph, i.e., the number
            of edges that enter or leave the egonet of v.
    """
    feature = snap.TIntV()
    
    feature_1 = Node.GetOutDeg()
    
    neighbor_list = list(Node.GetOutEdges())     #注意：NI.GetOutEdges()是返回节点邻居的序号！
    NIdV = snap.TIntV()
    for item in neighbor_list:
        NIdV.Add(item)
    NIdV.Add(Node.GetId())
    SubGraph = snap.GetSubGraph(Graph, NIdV)    #注意：subgraph没有算半条边！
    feature_2 = snap.CntUniqUndirEdges(SubGraph)
    
    Graph_copy = snap.ConvertGraph(type(Graph), Graph)  ##复制Graph
    snap.DelNodes(Graph_copy, NIdV)
    feature_3 =  Graph.GetEdges() - snap.CntUniqUndirEdges(SubGraph) - snap.CntUniqUndirEdges(Graph_copy)
    
    feature.Add(feature_1)
    feature.Add(feature_2)
    feature.Add(feature_3)
    
    return feature


def calcCosSimilarity(feature_u, feature_v):
    xy, xx, yy = 0, 0, 0
    for i in range(feature_u.Len()):
        xy += feature_u[i] * feature_v[i]
        xx += feature_u[i] ** 2
        yy += feature_v[i] ** 2
    
    if xx == 0. or yy == 0.:
        return 0. 
    return xy / (np.sqrt(xx) * np.sqrt(yy))



def calcTopkSimilarity(Node, Graph, k=5):
    res = snap.TIntV()
    dict_sim = {}
    feature_u = calcFeaturesNode(Node, Graph)
    for NI in Graph.Nodes():
        feature_v = calcFeaturesNode(NI, Graph)
        sim = calcCosSimilarity(feature_u, feature_v)
        dict_sim[NI.GetId()] = sim
    
    res_list = sorted(dict_sim.items(),key = lambda x:x[1],reverse = True)[1:k+1]
    res_list = list(zip(*res_list))[0]
    print (res_list)
    
    for item in res_list:
        res.Add(item)
        
    return res
    

def calcRecursiveFeaturesNode(Node, Graph, k=2):
    if k <= 0:
        return np.array(list(calcFeaturesNode(Node, Graph)))
 
    feature = np.zeros(3**k,)
    for NI in Node.GetOutEdges():
        feature_1 =  calcRecursiveFeaturesNode(Graph.GetNI(NI), Graph, k-1)
        feature += feature_1
    
    return np.concatenate((np.array(list(calcRecursiveFeaturesNode(Node, Graph, k-1))), feature / feature.shape[0], feature))



def calcTopkRecursiveSimilarity(Node, Graph, k=5):
    res = snap.TIntV()
    dict_sim = {}
    feature_u = calcRecursiveFeaturesNode(Node, Graph)
    for NI in Graph.Nodes():
        feature_v = calcRecursiveFeaturesNode(NI, Graph)
        if np.linalg.norm(feature_u)==0 or np.linalg.norm(feature_v)==0:
            sim = 0.
        else:
            sim = np.dot(feature_u, feature_v.T)/np.linalg.norm(feature_u)/np.linalg.norm(feature_v)
        dict_sim[NI.GetId()] = sim
    
    res_list = sorted(np.nan_to_num(list(dict_sim.items())),key = lambda x:x[1],reverse = True)[1:k+1]
    ##去掉Nan, np.nan_to_num()使用0代替数组x中的nan元素，使用有限的数字代替inf元素
    res_list = list(zip(*res_list))[0]
    res_list = list(map(int, res_list))
    print (res_list)
    
    for item in res_list:
        res.Add(item)
        
    return res


def getSubgraph(Node, Graph, hop=3):
    NodeVecAll = snap.TIntV()
    
    for i in range(1, hop+1):  
        NodeVec = snap.TIntV()
        snap.GetNodesAtHop(Graph, Node.GetId(), i, NodeVec, False)
        for item in NodeVec:
            NodeVecAll.Add(item)
    NodeVecAll.Add(Node.GetId())
    
    SubGraph = snap.GetSubGraph(Graph, NodeVecAll)
    return SubGraph


