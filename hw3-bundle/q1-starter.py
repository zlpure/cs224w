import snap
import numpy as np
import matplotlib.pyplot as plt

def load_graph(name):
    '''
    Helper function to load graphs.
    Use "epinions" for Epinions graph and "email" for Email graph.
    Check that the respective .txt files are in the same folder as this script;
    if not, change the paths below as required.
    '''
    if name == "epinions":
        G = snap.LoadEdgeList(snap.PNGraph, "soc-Epinions1.txt", 0, 1)
    elif name == 'email':
        G = snap.LoadEdgeList(snap.PNGraph, "email-EuAll.txt", 0, 1)   
    else: 
        raise ValueError("Invalid graph: please use 'email' or 'epinions'.")
    return G

def q1_1():
    '''
    You will have to run the inward and outward BFS trees for the 
    respective nodes and reason about whether they are in SCC, IN or OUT.
    You may find the SNAP function GetBfsTree() to be useful here.
    '''
    
    ##########################################################################
    #TODO: Run outward and inward BFS trees from node 2018, compare sizes 
    #and comment on where node 2018 lies.
    G = load_graph("email")
    #Your code here:
    BfsTree_out = snap.GetBfsTree(G, 2018, True, False)
    BfsTree_in = snap.GetBfsTree(G, 2018, False, True)
    out_, in_ = [], []
    for i in BfsTree_out.Nodes():
        out_.append(i.GetId())
    for i in BfsTree_in.Nodes():
        in_.append(i.GetId()) 
    print ('OUT, IN: ', len(out_), len(in_))
      
    
    ##########################################################################
    
    ##########################################################################
    #TODO: Run outward and inward BFS trees from node 224, compare sizes 
    #and comment on where node 224 lies.
    G = load_graph("epinions")
    #Your code here:
    BfsTree_out = snap.GetBfsTree(G, 224, True, False)
    BfsTree_in = snap.GetBfsTree(G, 224, False, True)
    out_, in_ = [], []
    for i in BfsTree_out.Nodes():
        out_.append(i.GetId())
    for i in BfsTree_in.Nodes():
        in_.append(i.GetId()) 
    print ('OUT, IN: ', len(out_), len(in_))
      
    ##########################################################################

    print ('2.1: Done!\n')


def q1_2():
    '''
    For each graph, get 100 random nodes and find the number of nodes in their
    inward and outward BFS trees starting from each node. Plot the cumulative
    number of nodes reached in the BFS runs, similar to the graph shown in 
    Broder et al. (see Figure in handout). You will need to have 4 figures,
    one each for the inward and outward BFS for each of email and epinions.
    
    Note: You may find the SNAP function GetRndNId() useful to get random
    node IDs (for initializing BFS).
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    G = load_graph("email")
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    out_num, in_num = [], []
    for i in range(0,100):
        NId = G.GetRndNId(Rnd)
        BfsTree_out = snap.GetBfsTree(G, NId, True, False)
        BfsTree_in = snap.GetBfsTree(G, NId, False, True)
        out_list = [i.GetId() for i in BfsTree_out.Nodes()]
        in_list = [i.GetId() for i in BfsTree_in.Nodes()]
        out_num.append(len(out_list))
        in_num.append(len(in_list))
        
        
    G = load_graph("epinions")
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    out_num_1, in_num_1 = [], []
    for i in range(0,100):
        NId = G.GetRndNId(Rnd)
        BfsTree_out = snap.GetBfsTree(G, NId, True, False)
        BfsTree_in = snap.GetBfsTree(G, NId, False, True)
        out_list = [i.GetId() for i in BfsTree_out.Nodes()]
        in_list = [i.GetId() for i in BfsTree_in.Nodes()]
        out_num_1.append(len(out_list))
        in_num_1.append(len(in_list))
    
    out_num.sort()
    in_num.sort()
    out_num_1.sort()
    in_num_1.sort()
    plt.subplot(221) 
    x = np.linspace(0, 1, num=len(out_num))
    plt.plot(x, out_num)
    plt.title('Reachability using outlinks in email')
    plt.xlabel('frac. of starting nodes')
    plt.ylabel('number of nodes reached')
    plt.subplot(222) 
    x = np.linspace(0, 1, num=len(in_num))
    plt.plot(x, in_num)
    plt.title('Reachability using inlinks in email')
    plt.xlabel('frac. of starting nodes')
    plt.ylabel('number of nodes reached')
    plt.subplot(223) 
    x = np.linspace(0, 1, num=len(out_num_1))
    plt.plot(x, out_num_1)
    plt.title('Reachability using outlinks in epinions')
    plt.xlabel('frac. of starting nodes')
    plt.ylabel('number of nodes reached')
    plt.subplot(224) 
    x = np.linspace(0, 1, num=len(in_num_1))
    plt.plot(x, in_num_1)
    plt.title('Reachability using inlinks in epinions')
    plt.xlabel('frac. of starting nodes')
    plt.ylabel('number of nodes reached')
      
    ##########################################################################
    print ('2.2: Done!\n')

def q1_3():
    '''
    For each graph, determine the size of the following regions:
        DISCONNECTED
        IN
        OUT
        SCC
        TENDRILS + TUBES
        
    You can use SNAP functions GetMxWcc() and GetMxScc() to get the sizes of 
    the largest WCC and SCC on each graph. 
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    G = load_graph("email")
    MxWcc = snap.GetMxWcc(G)
    disconnected = G.GetNodes() - len(list(MxWcc.Nodes()))
    MxScc = snap.GetMxScc(G)
    scc = len(list(MxScc.Nodes()))
    BfsTree_out = snap.GetBfsTree(G, 22, True, False)
    BfsTree_in = snap.GetBfsTree(G, 22, False, True)
    in_ = len(list(BfsTree_in.Nodes())) - scc
    out_ = len(list(BfsTree_out.Nodes())) - scc
    tendrils = G.GetNodes() - disconnected - scc - in_ - out_ 
    
    print ('DISCONNECTED, IN, OUT, SCC, TENDRILS + TUBES: ', disconnected, in_, out_, scc, tendrils)
        
    
    G = load_graph("epinions")
    MxWcc = snap.GetMxWcc(G)
    disconnected = G.GetNodes() - len(list(MxWcc.Nodes()))
    MxScc = snap.GetMxScc(G)
    scc = len(list(MxScc.Nodes()))
    BfsTree_out = snap.GetBfsTree(G, 22, True, False)
    BfsTree_in = snap.GetBfsTree(G, 22, False, True)
    in_ = len(list(BfsTree_in.Nodes())) - scc
    out_ = len(list(BfsTree_out.Nodes())) - scc
    tendrils = G.GetNodes() - disconnected - scc - in_ - out_ 
    
    print ('DISCONNECTED, IN, OUT, SCC, TENDRILS + TUBES: ', disconnected, in_, out_, scc, tendrils)
    
    
    ##########################################################################
    print ('2.3: Done!\n' )

def q1_4():
    '''
    For each graph, calculate the probability that a path exists between
    two nodes chosen uniformly from the overall graph.
    You can do this by choosing a large number of pairs of random nodes
    and calculating the fraction of these pairs which are connected.
    The following SNAP functions may be of help: GetRndNId(), GetShortPath()
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    G = load_graph("email")
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    num = 0
    NId_s = 22
    for i in range(0, 500):
        NId_e = G.GetRndNId(Rnd)
        NIdToDistH = snap.TIntH()
        shortestPath = snap.GetShortPath(G, NId_s, NIdToDistH, True)
        if NId_e in list(NIdToDistH):
            num += 1
    print (num / 500.)
    
    G = load_graph("epinions")
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    num = 0
    NId_s = 22
    for i in range(0, 500):
        NId_e = G.GetRndNId(Rnd)
        NIdToDistH = snap.TIntH()
        shortestPath = snap.GetShortPath(G, NId_s, NIdToDistH, True)
        if NId_e in list(NIdToDistH):
            num += 1
    print (num / 500.)
    
   
    ##########################################################################
    print ('2.4: Done!\n')
    
if __name__ == "__main__":
    q1_1()
    q1_2()
    q1_3()
    q1_4()
    print ("Done with Question 2!\n")