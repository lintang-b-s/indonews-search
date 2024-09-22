import vbcode


def encode_postings_list(posting_lists):
    """encode posting lists ke stream of bytes agar bisa disimpan di file
    
    Params
    ------ 
    posting_lists: List[int]
        list of docIDs

    Returns
    ------
    bytes
        bytearray berisi integer docIds di dalam posting_list
    """
    return vbcode.encode(posting_lists)

def decode_postings_list(encoded_postings_list):
    """decode posting lists dari stream of bytes
    

    Params
    ------
    encoded_postings_list: bytes
        bytearray berisi posting lists hasil dari encoding

    Returns
    ------
    List[int]
        posting_lists yang sudah didecode/list of docIDs
    """
    return vbcode.decode(encoded_postings_list)
