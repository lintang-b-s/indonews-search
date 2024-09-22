import os 
from fts.utils import *
import pickle as pkl 

class InvertedIndex:
    """
    Class ini menimplemntasikan method untuk read dan write inverted index ke disk

    Attr
    -------
    postings_dict = dictionary[term](start_position_in_index_file, 
                                                number_of_postings_in_list,
                                               length_in_bytes_of_postings_list)
        memetakan termID ke metadata dari posting lists tersebut yang digunakan 
        untuk reading dan writing postings ke disk. start_position_in_index_file adalah posisi awal
        dari posting lists di dalam index file (digunakan saat file.seek(start_position_in_index_file)).
        number_of_postings_in_list adalah jumlah docIDs didalam postings.
        length_in_bytes_of_postings_list adalah ukuran postings di dalam index file (digunakan untuk 
        read posting dari index file -> file.read(length_in_bytes_of_postings_list) dari posisi start_position_in_index_file
        -> file.seek(start_position_in_index_file)).
        
        
    terms: List[int]
        list of termID
    """
    def __init__(self, index_name,  directory=''):
      """
      Params
      ------
      index_name: string
          nama index file
      postings_encoding: class
          class untuk encoding/decoding
      directory: 
          directory untuk menyimpan index files
          
      """
      self.index_file_path = os.path.join(directory, index_name+'.index')
      self.metadata_file_path = os.path.join(directory, index_name+'.dict')

   
      self.directory = directory

      self.postings_dict = {}
      self.doc_term_count_dict = {} # docID: count term di inverted index ini
      self.terms = []

    def open_writer(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self
        
    def __enter__(self):
        """Membuka file inverted index dan file metadata , serta load metadata  postings_dict, terms"""
        self.index_file = open(self.index_file_path, 'rb+')

        # load posting list dan terms dari file metadata
        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms, self.doc_term_count_dict = pkl.load(f)
            self.term_iter = self.terms.__iter__()

        return self
    

    def __exit__(self, exception_type, exception_value, traceback):
        """"close file inverted index  dan save file inverted index"""

        self.index_file.close()

        with open(self.metadata_file_path, 'wb') as f:
            pkl.dump([self.postings_dict, self.terms, self.doc_term_count_dict], f)
            
    def exit(self):
        self.index_file.close()

        with open(self.metadata_file_path, 'wb') as f:
            pkl.dump([self.postings_dict, self.terms, self.doc_term_count_dict], f)
        

    def __getitem__(self, key):
        return self._get_postings_list(key)

    def _get_postings_list(self, term):
        """"
        mendapatkan postinglists dari term

        Params
        -----
        Int
            termID 
        """
        try:
            start_position, n_postings, length_in_bytes = self.postings_dict[term]
            self.index_file.seek(start_position)
            return decode_postings_list(self.index_file.read(length_in_bytes))
        except KeyError:
            return []

    def add_doc_term_count(self, docID, term_count):
        self.doc_term_count_dict[docID] = term_count

    def append(self, term, postings_list):
        """
        append term dan posting lists ke index file (posisi di block paling akhir di file)
    
        posting_dict memetakan termID ke tuple, isi posting_dict:
        (start_position_in_index_file,
               number_of_postings_in_list,
               length_in_bytes_of_postings_list)
        """
        encoded_postings_list = encode_postings_list(postings_list) # bytearrray dari posting_lists
        start_position_in_index_file = self.index_file.seek(0, 2) # posisi paling akhir di file
        length_in_bytes_of_postings_list = self.index_file.write(encoded_postings_list) # write bytearray posting_lists ke file index
        self.terms.append(term) # tambahkan term ke list of terms
        self.postings_dict[term] = (start_position_in_index_file, len(postings_list), length_in_bytes_of_postings_list ) # simpan metadata dari term

    





class InvertedIndexIterator(InvertedIndex):
  """
        iterator untuk setiap block inverted index.
        read file dari disk hanya dilakukan saat process merge (ketika heapq.merge(*indices) 
            method __next__ dari iterator akan dipanggil dan read posting lists dari term ke disk).
       setiap iterasi, yang diload ke memory hanya satu posting list dari disk. tujuannya agar posting lists yang 
       di load ke memory tidak terlalu besar/tidak semua posting lists yang di 
        disk di load ke memory. 
  """
  def enter(self):
    super().__enter__()
    self.curr_term_pos = 0 
    return self 


  def __iter__(self):
   
    return self

  def __next__(self):
    """
        return  pasangan (termID, posting_lists) selanjutnya
        

        read satu posting lists dari term dari index file/disk

        Returns
        -----
        Int
            termID yang di read
        List[Int]
            posting lists dari termID
        
    """
    if self.curr_term_pos < len(self.terms):
        term = self.terms[self.curr_term_pos]
        self.curr_term_pos += 1
        start_position, n_postings, length_in_bytes = self.postings_dict[term]
        self.index_file.seek(start_position)
        postings_list = decode_postings_list(self.index_file.read(length_in_bytes))
        return term, postings_list
    else:
        raise StopIteration


  def exit(self):
      """
        close file index
      """
      super().exit()       

  def exit_and_remove(self):
      self.index_file.close() 
      
      if hasattr(self, 'delete_upon_exit') and self.delete_upon_exit:
          os.remove(self.index_file_path)
          os.remove(self.metadata_file_path)
      