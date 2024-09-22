import networkx as nx

class Slice:
    def __init__(self):
        """
        Initializes the Slice class. 
        Currently, no specific initialization is required.
        """
        pass

    # Function to extract a graph structure from a slice of blocks
    def extract_graph_from_slice(self, block_slice):
        """
        Converts a slice of blocks into a directed graph (DiGraph) representation.

        Args:
        block_slice (list): A list of blocks where each block is a dictionary 
                            containing block attributes and connections.

        Returns:
        networkx.DiGraph: A directed graph where nodes represent blocks and edges represent connections.
        """
        # Initialize an empty directed graph
        G = nx.DiGraph()

        # Dictionary to map SID (block IDs) to node indices in the graph
        sid_to_index = {}

        # Counter to assign unique indices to each block
        index_counter = 0

        # Set of all SIDs (block identifiers) in the current block slice
        all_sids = set(block['SID'] for block in block_slice)

        # List of attributes to extract from each block for node properties
        attributes = ['OutputSignals', 'OutDataTypeStr', 'UpperLimit', 'LowerLimit', 
                      'Function', 'SampleTime', 'Operator']

        # Assume the last block in the block_slice is the "main block"
        mainBlock = block_slice[-1]

        def add_node_with_attributes(sid, block):
            """
            Adds a node to the graph with the specified SID and block attributes.

            Args:
            sid (str): The unique SID (block identifier).
            block (dict): The block data containing attributes to assign to the node.
            """
            # Get the index for the node corresponding to the given SID
            node_index = sid_to_index[sid]

            # Initialize a dictionary to store node attributes
            node_attributes = {'SID': sid, 'BlockType': block['BlockType']}

            # If the block has a 'PropertyDict', add selected attributes to the node
            if 'PropertyDict' in block:
                for attr in attributes:
                    # Extract specific attributes from 'PropertyDict' or set them to empty if not found
                    node_attributes[attr] = block['PropertyDict'].get(attr, '')

            # Mark the "main" block (the last block in the slice) with additional attributes
            if block['SID'] == mainBlock['SID']:
                node_attributes['is_main'] = True  # Mark as main block
                node_attributes['SliceSize'] = len(block_slice)  # Store the size of the block slice

            # Add the node with its attributes to the graph
            G.add_node(node_index, **node_attributes)

        # Iterate over each block in the block slice
        for block in block_slice:
            destinationIDs = block['DestinationBlockSIDs']  # Blocks this block sends data to
            sourceIDs = block['SourceBlockSIDs']  # Blocks that send data to this block
            
            # Assign a unique index to the current block if it's not already indexed
            if block['SID'] not in sid_to_index:
                sid_to_index[block['SID']] = index_counter
                index_counter += 1

            # Add the current block as a node with attributes
            add_node_with_attributes(block['SID'], block)

            # Add edges between the current block and its destination blocks
            for destID in destinationIDs:
                if destID in all_sids and destID not in sid_to_index:
                    # Assign an index to destination blocks if not already indexed
                    sid_to_index[destID] = index_counter
                    index_counter += 1
                # Add an edge from the current block to the destination block
                G.add_edge(sid_to_index[block['SID']], sid_to_index[destID])

            # Add edges between the source blocks and the current block
            for sourceID in sourceIDs:
                if sourceID in all_sids and sourceID not in sid_to_index:
                    # Assign an index to source blocks if not already indexed
                    sid_to_index[sourceID] = index_counter
                    index_counter += 1
                # Add an edge from the source block to the current block
                G.add_edge(sid_to_index[sourceID], sid_to_index[block['SID']])

        # Return the constructed directed graph
        return G
