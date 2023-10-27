
import networkx as nx
import matplotlib.pyplot as plt   
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from .path import get_varnames

def draw_DAG(model,figsize=(10,10),node_size=20_000,font_size=12,dpi=100,order=None,filename=None):

    # a. re-format blocks
    all_inputs = set()
    for varname in model.shocks+model.unknowns:
        all_inputs.add(varname)

    blockdict = {}
    for blockstr in model.blocks:

        if blockstr == 'hh':
            
            name = 'hh'
            inputs = [varname for varname in model.inputs_hh_all]

            for varname in inputs:
                assert varname in all_inputs, f'{varname} not defined before hh block'

            inputs += [f'{varname.upper()}_hh' for varname in model.outputs_hh]

        else:
            
            name = blockstr.split('.')[1]
            inputs = get_varnames(blockstr)   

        outputs = [varname for varname in inputs if not varname in all_inputs]
        
        name += '\n['
        for varname in outputs:
            if varname in model.targets:
                name += f'{varname},\n'
        if name[-1] == '[':
            name = name[:-1]
        else:
            name = name[:-2] + ']'

        blockdict[blockstr] = (name,inputs,outputs)

        for varname in outputs:
            all_inputs.add(varname)

    # b. edges
    edges = []

    for left in model.shocks+model.unknowns:
        for blockstr in model.blocks:
            name,inputs,_outputs = blockdict[blockstr]
            if left in inputs:
                edge = (left,name)  
                edges.append(edge)

    for blockstr_left in model.blocks:
        name_left,_inputs_left,outputs_left = blockdict[blockstr_left]
        for blockstr_right in model.blocks:    
            if blockstr_left == blockstr_right: continue
            name_right,inputs_right,_outputs_right = blockdict[blockstr_right]
            for left in outputs_left:
                if left in inputs_right:
                    edge = (name_left,name_right)
                    if not edge in edges:
                        edges.append(edge)

    # c. network
    graph = nx.DiGraph()
    graph.add_edges_from(edges)    

    # d. plot
    fig = plt.figure(figsize=figsize,dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    
    ax.grid(False)
    ax.axis('off')

    blocks = [g for g in graph.nodes if not g in model.shocks+model.unknowns]
    
    if order is None:
        nlist = None
    else:
        nlist = []
        for name in order:
            if name == 'blocks':
                nlist.append(blocks)
            elif name == 'shocks':
                nlist.append(model.shocks)
            elif name == 'unknowns':
                nlist.append(model.unknowns)
            else:
                raise ValueError('{name} in order not allowed')

    pos = nx.shell_layout(graph,nlist=nlist)

    nx.draw_networkx_nodes(graph,pos=pos,ax=ax,node_size=node_size,node_color=colors[0],
        node_shape='s',alpha=0.5,nodelist=model.unknowns,margins=0.1)
    nx.draw_networkx_nodes(graph,pos=pos,ax=ax,node_size=node_size,
        node_color=colors[1],node_shape='s',alpha=0.5,nodelist=model.shocks,margins=0.1)
    nx.draw_networkx_nodes(graph,pos=pos,ax=ax,node_size=node_size,
        node_color=colors[2],node_shape='o',alpha=0.5,nodelist=blocks,margins=0.1)

    nx.draw_networkx_edges(graph,pos=pos,ax=ax,node_size=node_size)
    nx.draw_networkx_labels(graph,pos=pos,ax=ax,font_size=font_size)    
    
    fig.tight_layout()

    if not filename is None: fig.savefig(filename)