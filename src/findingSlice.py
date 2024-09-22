import networkx as nx

class Slice:
    def __init__(self):
        pass

    def extract_graph_from_slice(self, block_slice):
        G = nx.DiGraph()
        sid_to_index = {}
        index_counter = 0
        all_sids = set(block['SID'] for block in block_slice)

        attributes = ['OutputSignals', 'OutDataTypeStr', 'UpperLimit', 'LowerLimit', 
                      'Function', 'SampleTime', 'Operator']
        mainBlock = block_slice[-1]

        def add_node_with_attributes(sid, block):
            node_index = sid_to_index[sid]
            node_attributes = {'SID': sid, 'BlockType': block['BlockType']}
            if 'PropertyDict' in block:
                for attr in attributes:
                    node_attributes[attr] = block['PropertyDict'].get(attr, '')
            if block['SID'] == mainBlock['SID']:
                node_attributes['is_main'] = True
                node_attributes['SliceSize'] = len(block_slice)
            G.add_node(node_index, **node_attributes)

        for block in block_slice:
            destinationIDs = block['DestinationBlockSIDs']
            sourceIDs = block['SourceBlockSIDs']
            
            if block['SID'] not in sid_to_index:
                sid_to_index[block['SID']] = index_counter
                index_counter += 1
            
            add_node_with_attributes(block['SID'], block)

            for destID in destinationIDs:
                if destID in all_sids and destID not in sid_to_index:
                    sid_to_index[destID] = index_counter
                    index_counter += 1
                G.add_edge(sid_to_index[block['SID']], sid_to_index[destID])
            
            for sourceID in sourceIDs:
                if sourceID in all_sids and sourceID not in sid_to_index:
                    sid_to_index[sourceID] = index_counter
                    index_counter += 1
                G.add_edge(sid_to_index[sourceID], sid_to_index[block['SID']])

        return G
