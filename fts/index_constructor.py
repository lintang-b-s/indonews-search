import numpy as np
import time 
import heapq
from collections import defaultdict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from fts.mapper import IdMap
from fts.parser import *
from fts.index import *
from collections import OrderedDict

import sys
import re



nltk.download('popular')
nltk.download('punkt_tab')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(nltk.corpus.stopwords.words('indonesian'))


class DynamicSIPMI_BSBIIndexer:
  """Blocked sort-based Indexing & Dynamic Indexing & tf idf &  pembuatan inverted index
  SIPMI_BSBI:
  1. mensegmentasi seluruh dokumen di dalam csv menjadi 11 block/bagian.
  2. sorting pasangan termID-pairID setiap block berdasarkan termID lalu sort berdasarkan pairID dan buat posting lists .
  3. menyimpan sorted posting lists ke index file di disk.
  4. merge semua block inverted index yang ada di intermediate_indices dan write ke disk (hasil akhir inverted index).
  ref:  https://nlp.stanford.edu/IR-book/pdf/04const.pdf
  ref2: https://web.stanford.edu/class/cs276/19handouts/lecture2-intro-boolean-6per.pdf
  ref3: https://nlp.stanford.edu/IR-book/pdf/06vect.pdf
  

  Attrs
  ------
  term_id_map: IdMap
      untuk mapping term dari string ke termID
  doc_id_map: IdMap
    untuk mapping doc dari title string ke docID
  file_path: str
      path ke file data input csv yang akan di index
  output_dir: str
      path directory output untuk menyimpan setiap block inverted index dan merged index
  index_name: str
      nama merged index (file index berisi semua posting lists dari data input)
  postings_encoding: 
      encoding untuk menyimpan postings list ke disk (default nya UncompressedPostings).
  docWordCount: dict{term: term}
      menyimpan jumlah kata di dalam dokumen
  in_memory_indices: dict{term: postings_list}
    auxilary inverted index di dalam memory, setiap doc baru diindex ke database, indexnya diinsert disini.
    setelah ukuran postings_list in_memory_indices > 1e8, merge dengan inverted index di indices 
  indices: set
    list of inverted index file selain SIPMI_BSBI_Lintang_main (di dalam disk)
  max_dynamic_posting_list_size: int
    ukuran maksimal jumlah posting list  di in_memory_indices
  invalidation_bit_vector: [int]
    buat nandain docID yang udah di delete dari inverted idnex. index dari array == docID yang sudah didelete dari inverted index.
    ref: "Deletions are stored in an invalidation bit vector. We can then filter out deleted documents before returning the search result. Documents are updated by deleting and reinserting them."
    from: https://nlp.stanford.edu/IR-book/html/htmledition/dynamic-indexing-1.html

  """
  def __init__(self, file_path, output_dir, index_name = "SIPMI_BSBI_Lintang_main", inverted_index_buffer_size = 1e8
                 ):
    self.term_id_map = IdMap()
    self.doc_id_map = IdMap()
    self.file_path = file_path
    self.output_dir = output_dir
    self.index_name = index_name
    self.docLen = lenDocs
    self.intermediate_indices = []
    self.docWordCount =  {}
    self.in_memory_indices = {}
    self.indexes = set()
    
    self.max_dynamic_posting_list_size =  inverted_index_buffer_size #1e8   # 1e8 postings di in_memory_indices . 1e8 * 32 bit (int) = 400mb max size in-memory inverted index, sebelum diwrite ke disk
    self.initialization()


  
      

  def initialization(self):
    dynamic_index_filename = "DynamicSIPMI_BSBI_Lintang_"
    print("initializing: meload data dari database file...")
    for (_, _, filenames) in os.walk("./output_dir"):
        if len(filenames) != 0:
                self.load()
        else:
            self.invalidation_bit_vector = []
        for filename in filenames:
            
            if dynamic_index_filename in filename and filename.endswith('.dict'):
                i = re.findall(r'\d+', filename)
                i = i[0]
                self.indexes.add(int(i))
                index_file = "DynamicSIPMI_BSBI_Lintang_" + str(i)
                with InvertedIndex(index_file, self.output_dir) as curr_idx:
                    for docID, termCount in curr_idx.doc_term_count_dict.items():
                        self.docWordCount[docID] = termCount
    
    print("initializing: selesai")


  def save(self):
      """menyimpan doc_id_map dan term_id_map ke output directory"""


      with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
          pkl.dump(self.term_id_map, f)
      with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
          pkl.dump(self.doc_id_map, f)
      with open(os.path.join(self.output_dir, 'doc_word_count.dict'), 'wb') as f:
          pkl.dump((self.docWordCount, self.invalidation_bit_vector ), f)
    

  def load(self):
      """Load doc_id_map and term_id_map dari output directory"""

      with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
          self.term_id_map = pkl.load(f)
      with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
          self.doc_id_map = pkl.load(f)
      with open(os.path.join(self.output_dir, 'doc_word_count.dict'), 'rb') as f:
          self.docWordCount, self.invalidation_bit_vector = pkl.load(f)

  def merge_index(self, Z, curr_index):
      """
      Z: dict{term: postings_list}
      token_index: InvertedIndexIterator
      """
      
      for token, postings_list in heapq.merge(curr_index):
         # https://eecs485staff.github.io/p4-mapreduce/heapq.html
         if token in Z:
             for new_postings_list in postings_list:
                Z[token].append(new_postings_list)
             Z[token] = sorted(Z[token])
         else:
             Z[token] = postings_list
      return Z
  
  

  def index_doc_to_inmemory_indices(self, doc: str, title: str):
      """
        invert doc jadi posting list & append ke in_memory inverted index
        ref: https://web.stanford.edu/class/cs276/19handouts/lecture2-intro-boolean-6per.pdf
      """
      
      term_doc_pairs = []
      doc_id = self.doc_id_map[title]
      doc = stemmer.stem(doc) # stemming
      words = nltk.word_tokenize(doc) # tokenisasi
      self.docWordCount[doc_id] = len(words) # simpan word count dokumen untuk hitung tf
      
      for word in words:
        word = word.lower()  # mengubah kata ke lowercase
        if word not in stop_words: # stopword removal
            term_id = self.term_id_map[word]
            term_doc_pairs.append([term_id, doc_id])
      
      term_doc_dict =  defaultdict(list)
      for t,d in term_doc_pairs:
        term_doc_dict[t].append(d)

      for t in sorted(term_doc_dict.keys()):
        unsorted_posting_list = term_doc_dict[t]

        sorted_posting_list = sorted(unsorted_posting_list)

        self.update_idf(t, len(sorted_posting_list))

        if t in self.in_memory_indices:
            for posting in sorted_posting_list:
                self.in_memory_indices[t].append(posting)
            self.in_memory_indices[t] = sorted(self.in_memory_indices[t])
        else:
            self.in_memory_indices[t] = sorted_posting_list
      in_memory_indices_size = 0
      for token in self.in_memory_indices:
          in_memory_indices_size += len(self.in_memory_indices[token])

     
      return in_memory_indices_size    

  
      

  def write_indices_to_disk(self, indices, index_writer):
        """
        indices: dict{term: postings_list}
            write postings list dari inverted index ke file di dalam disk
        index_writer: InvertedIndex
          untuk menulis ke inverted index file
        """
        doc_term_counter = dict()
        for token in sorted(indices.keys()):
          sorted_postings_list = sorted(indices[token])
          index_writer.append(token, sorted_postings_list)

          for doc_id in sorted_postings_list:
              if doc_id not in doc_term_counter:
                doc_term_counter[doc_id] = 1
              else:
                doc_term_counter[doc_id] += 1
        for docID, termCount in doc_term_counter.items():
            index_writer.add_doc_term_count(docID, termCount)

        self.save() #write term_id_map & doc_id_map yang mengandung term di new indexed docs

        
  def close(self):
      print("closing database... and writing in-memory inverted indexes to disk")
      for i in range(0, sys.maxsize ):
          curr_index_name = "DynamicSIPMI_BSBI_Lintang_" + str(i)
          if i not in self.indexes :
              # write in_memory inverted index ke disk
              # write index_i ke disk
              if len(self.in_memory_indices) != 0:
                index_i = self.in_memory_indices
                self.indexes.add(int(i))
                index_writer = InvertedIndex(curr_index_name, directory=self.output_dir).open_writer()
                self.write_indices_to_disk(index_i, index_writer)
                index_writer.exit()
              break
          elif  os.path.getsize('./output_dir/{}.index'.format(curr_index_name))/1e6 < 400:
              if len(self.in_memory_indices) != 0:
                
                index_i  = InvertedIndexIterator(curr_index_name,
                                        directory=self.output_dir).enter() # open index_i file
              
                Zi = self.merge_index(self.in_memory_indices, index_i) # merge inverted index index_i dengan inverted index Zi
                index_i.exit()

                index_writer = InvertedIndex(curr_index_name, directory=self.output_dir).open_writer()
                self.write_indices_to_disk(Zi, index_writer)
                
                index_writer.exit()
              break
      self.save()
            
  def lMergeAddToken(self, doc, title):
      """
      algoritma buat dynamic indexing.
      https://nlp.stanford.edu/IR-book/html/htmledition/dynamic-indexing-1.html
      dipanggil setiap kali dokumen baru di index ke database.

      save new indexed doc di in-memory index, baru setelah size in-memory index > 400mb save ke disk & clear in-memory index
      ref: "If there is a requirement that new documents be included quickly, one solution is to maintain two indexes: a large main index and a small auxiliary index that stores new documents. The auxiliary index is kept in memory. Searches are run across both indexes and results merged."
      from: https://nlp.stanford.edu/IR-book/html/htmledition/dynamic-indexing-1.html
      
      in_memory_indices:  dict{term: postings_list}
      doc: str
        content document baru yang ingin diindex

      title: str
        title dari dokument yang ingin diindex
      """
      in_memory_indices_size = self.index_doc_to_inmemory_indices( doc, title)
      self.invalidation_bit_vector.append(self.doc_id_map[title])
      self.total_doc_num += 1
      if in_memory_indices_size >= self.max_dynamic_posting_list_size:
        index_i = None
        Zi = OrderedDict(sorted(self.in_memory_indices.items()))
        for i in range(0, sys.maxsize ):
          curr_index_name = "DynamicSIPMI_BSBI_Lintang_" + str(i)
          if i in self.indexes:
              
              index_i  = InvertedIndexIterator(curr_index_name,
                                        directory=self.output_dir).enter() # open index_i file
              
              Zi = self.merge_index(Zi, index_i) # merge inverted index index_i dengan inverted index Zi

              self.indexes.remove(int(i))

              index_i.exit_and_remove()
          else:
              index_i = Zi
              self.indexes.add(int(i))

              # write index_i ke disk
              index_writer = InvertedIndex(curr_index_name, directory=self.output_dir).open_writer()
              self.write_indices_to_disk(index_i, index_writer)
              index_writer.exit()
              break
        self.in_memory_indices = {}
    
  
    

             

  def inverted_index_from_doc(self, block, i):
     # membuat inverted index untuk setiap block 
     td_pairs = self.parse_block(block)
     index_id = 'index_'+str(i) # membuat inverted index setiap block
      

     index = InvertedIndex(index_id, directory=self.output_dir).open_writer()
        
        # membuka file index_id dan write posting list ke dalam file tersebut
        # setelah itu ke block berikutnya dan close file index_id
     self.invert_write(td_pairs, index)
     td_pairs = None

     self.intermediate_indices.append(index_id)
     index.exit()
     print("berhasil mengideks batch data ke-", i)

  
  def sipmi_parse_block(self, doc, title):
      """
        Parse satu block yang berisi list of documents dari csv menjadi termID-docID pairs
    
        Params
        ------
        doc: string
            content dokumen
        title: string
            title dokumen
        Returns
        ------
        List[tuple[Int, Int]]
            list semua termID, docID pairs
        
      """
      if type(doc) == float:
          return []
      term_doc_pairs = []
      doc_id = self.doc_id_map[title]
      doc = stemmer.stem(doc) # stemming
      words = nltk.word_tokenize(doc) # tokenisasi
      self.docWordCount[doc_id] = len(words) # simpan word count dokumen untuk hitung tf
            
      for word in words:
            word = word.lower()  # mengubah kata ke lowercase
            if word not in stop_words: # stopword removal
              term_id = self.term_id_map[word] # harusnya tidak perlu simpan term->termID kalau SIPMI, create separate dictionary utk setiap block index
              term_doc_pairs.append([term_id, doc_id])
      return term_doc_pairs

  def sipmi_invert(self, df):
      """
      indexing pakai single-pass in-memory indexing
      https://nlp.stanford.edu/IR-book/pdf/04const.pdf
      

      """
      dictionary=  dict()
      posting_size = 0
      block = 0
      
      for ind in df.index:
        
        term_doc_pairs = self.sipmi_parse_block(df["content"][ind], data["title"][ind])
        if len(term_doc_pairs) == 0:
            continue
        for term, doc_id in term_doc_pairs:
            postings_list = None
            if term not in dictionary:
                postings_list = []
                dictionary[term] = postings_list
            else:
                postings_list = dictionary[term]
            postings_list.append(doc_id)
            dictionary[term] = postings_list
            posting_size += 1

        if posting_size >= 10e5:
            posting_size = 0
            sorted_terms = sorted(list(dictionary.keys()))
            index_id = 'index_'+str(block) # membuat inverted index setiap block
            print("menyimpan ", index_id, " ke disk")
            index = InvertedIndex(index_id, directory=self.output_dir).open_writer()
            self.intermediate_indices.append(index_id)
            for t in sorted_terms:
                sorted_posting_list = sorted(dictionary[t])
                index.append(t, sorted_posting_list)
            block +=1
            dictionary=  dict()
            index.exit()
            
      # last block index
      sorted_terms = sorted(list(dictionary.keys()))
      index_id = 'index_'+str(block) # membuat inverted index setiap block
      print("menyimpan ", index_id, " ke disk")
      index = InvertedIndex(index_id, directory=self.output_dir).open_writer()
      self.intermediate_indices.append(index_id)
      for t in sorted_terms:
        sorted_posting_list = sorted(dictionary[t])
        index.append(t, sorted_posting_list)
      block +=1
      index.exit()
        



  def sipmi_index(self):
      print("indexing document di dalam corpus (News.csv)....")
      start = time.time()
      df =  pd.read_csv(self.file_path)
      self.sipmi_invert(df)

      self.save()
      merged_index  = InvertedIndex(self.index_name, directory=self.output_dir ).open_writer() 
          # membuka file index_name dan merge semua block inverted index ke merged_index
          # write merged_index ke file index_name
      indices = []
      try:
          for index_id in self.intermediate_indices:
              
                curr_idx = InvertedIndexIterator(index_id,
                                        directory=self.output_dir).enter()
                indices.append(curr_idx)
          self.merge(indices, merged_index)
      except Exception as e:
          print(f"An error occurred: {e}")
      finally:
        for indices in indices:
            indices.exit()      
        
      merged_index.exit()

      end = time.time()

      print(f"total waktu yang dibutuhkan untuk indexing seluruh dokumen: {(end-start)*10**3:.03f}ms")
      self.invalidation_bit_vector = [False] * len(self.docWordCount)


      
  def delete_doc(self, doc_id):
      """
      delete document dg doc_id dari inverted index
      ref: "Deletions are stored in an invalidation bit vector"
      https://nlp.stanford.edu/IR-book/html/htmledition/dynamic-indexing-1.html
      """
      temp_title = self.doc_id_map[doc_id]
      if temp_title == '':
          return 
      del self.doc_id_map[temp_title] 
      self.doc_id_map[doc_id] = ''

      self.invalidation_bit_vector[doc_id] = True


  def update_doc(self, doc_id, title, content):
      """
      update document dg doc_id di inverted index
      ref: " Documents are updated by deleting and reinserting them."
      https://nlp.stanford.edu/IR-book/html/htmledition/dynamic-indexing-1.html
      """
      self.delete_doc(doc_id)
      self.lMergeAddToken(content, title)

  def index(self):
      """
        indexing semua document di file csv menggunakan Block sorted-based indexing
        https://nlp.stanford.edu/IR-book/html/htmledition/blocked-sort-based-indexing-1.html
      """

      start = time.time()

      
      chunksize =  1434
      file_name = self.file_path.split("/")[-1]
      i = 0
      # read chunk of csv dengan setiap chunk berukuran 1434
  
      for block in pd.read_csv(self.file_path, chunksize=chunksize):
         # membuat inverted index untuk setiap block
        self.inverted_index_from_doc(block, i)
        i +=1    

      self.save()
      merged_index  = InvertedIndex(self.index_name, directory=self.output_dir ).open_writer() 
          # membuka file index_name dan merge semua block inverted index ke merged_index
          # write merged_index ke file index_name
      indices = []
      try:
          for index_id in self.intermediate_indices:
              
                curr_idx = InvertedIndexIterator(index_id,
                                        directory=self.output_dir).enter()
                indices.append(curr_idx)
          self.merge(indices, merged_index)
      except Exception as e:
          print(f"An error occurred: {e}")
      finally:
        for indices in indices:
            indices.exit()      
        
      merged_index.exit()

      end = time.time()

      print(f"total waktu yang dibutuhkan untuk indexing seluruh dokumen: {(end-start)*10**3:.03f}ms")
      self.invalidation_bit_vector = [False] * len(self.docWordCount)


  def merge(self, indices, merged_index):
        """
        merge semua blok inverted index menjadi satu index (merged_index)
        dan write merged_index ke file index (melalui method append() dari InvertedIndex

        Params
        ------
        indices: List[InvertedIndexIterator]
            list of InvertedIndexIterator, setiap elemen nya adalah satu block inverted index (hasil
            dari split dokumen menjadi 11 block inverted index).
        merged_index: InvertedIndex
            hasil akhir inverted index, dimana setiap block inverted index dimerge ke merged_index ini.
        """
        last_term = last_posting = None
        for curr_term, curr_postings in heapq.merge(*indices):
            # heapq.merge mereturn list of curr_term dan curr_postings yang sudah disort berdasarkan term dan postings_list
            # posting lists didapatkan dari read block of bytes ke index file yang ada didalam disk (method __next__ InvertedIndexIterator)
            # https://eecs485staff.github.io/p4-mapreduce/heapq.html
            if curr_term != last_term:
                if last_term:
                    last_posting = list(sorted(last_posting)) # sort previous posting lists berdasarkan docIDs
                    merged_index.append(last_term, last_posting) # write previous merged posting lists ke file index
                last_term = curr_term
                last_posting = curr_postings
            else:
                last_posting += curr_postings
        if last_term:
            last_posting = list(sorted(last_posting)) 
            merged_index.append(last_term, last_posting) # write merged posting lists ke file index


  def parse_block(self, block):
      """
        Parse satu block yang berisi list of documents dari csv menjadi termID-docID pairs
    
        Params
        ------
        block: pandas.Dataframe
           satu block dataframe (hasil dari read csv dengan ukuran chunksize)

        Returns
        ------
        List[tuple[Int, Int]]
            list semua termID, docID pairs
        
      """
      term_doc_pairs = []

      for ind in block.index:
        
        doc = block["content"][ind]
        if type(doc) != str:
          continue
        doc_id = self.doc_id_map[block["title"][ind]]
        doc = stemmer.stem(doc) # stemming
        words = nltk.word_tokenize(doc) # tokenisasi
        self.docWordCount[doc_id] = len(words) # simpan word count dokumen untuk hitung tf
          
        for word in words:
          word = word.lower()  # mengubah kata ke lowercase
          if word not in stop_words: # stopword removal
              term_id = self.term_id_map[word]
              term_doc_pairs.append([term_id, doc_id])
      return term_doc_pairs

  def invert_write(self, td_pairs, index):
    """
    Invert td_pairs menjadi posting lists .
     sorting berdasarkan termID lalu sort lagi berdasarkan docID.
    lalu simpan posting list ke file indeex melalui method append() dari InvertedIndex
    
     Params
     ------
     td_pairs: List[Tuple[Int,Int]]
         list semua termID, docID pairs
     index: InvertedIndex
         satu block inverted index 
    """
    term_doc_dict =  defaultdict(list)
    for t,d in td_pairs:
        term_doc_dict[t].append(d)
    for t in sorted(term_doc_dict.keys()):
      unsorted_posting_list = term_doc_dict[t]

      sorted_posting_list = sorted(unsorted_posting_list)
    
      index.append(t, sorted(sorted_posting_list))

  
      


  def build_idf(self ):
        """
        membuat dictionary inverse document frequency
        ref: https://web.stanford.edu/class/cs276/19handouts/lecture6-tfidf-6per.pdf

        """
        print("building idf dictionry....")
        try:
            with open("./output_dir/docs.dict", 'rb') as f:
                docs = pkl.load(f)
            self.total_doc_num = len(docs)
            self.idf = {}
            
            metadata_file_path = os.path.join("output_dir", "SIPMI_BSBI_Lintang_main"+'.dict')
            with open(metadata_file_path, 'rb') as f:
                postings_dict, terms, doc_term_count_dict = pkl.load(f)

            index_file_path = os.path.join("output_dir", "SIPMI_BSBI_Lintang_main"+'.index')
            index_file = open(index_file_path, 'rb+')
            self.df = {}

            # hitung df untuk setiap term di inverted index SIPMI_BSBI_Lintang_main
            for t_id, (start_position, _, length_in_bytes) in postings_dict.items():
                index_file.seek(start_position)
                postings_list = decode_postings_list(index_file.read(length_in_bytes))
                # idf = np.log10(self.total_doc_num) - np.log10(df+1)
                # self.idf[t_id] = idf
                self.df[t_id] = len(set(postings_list))
            index_file.close()

            # hitung df untuk setiap term di in memory inverted index
            for token_id, postings_list in self.in_memory_indices.items():
                if token_id not in self.df:
                    self.df[token_id] = len(set(postings_list))
                else:
                    self.df[token_id] += len(set(postings_list))
            
            # menghitung df untuk setiap term di setiap inverted index (yang didalam disk) di self.indexes
            for inverted_index in self.indexes:
                with InvertedIndex("DynamicSIPMI_BSBI_Lintang_"+str(inverted_index), directory=self.output_dir
                    ) as curr_index:
                        for t_id, (start_position, _, length_in_bytes) in curr_index.postings_dict.items():
                            curr_index.index_file.seek(start_position)
                            postings_list = decode_postings_list(curr_index.index_file.read(length_in_bytes))
                            if t_id not in self.df:
                                self.df[t_id] =  len(set(postings_list))
                            else:
                                self.df[t_id] +=  len(set(postings_list))

            for term_id, freq in self.df.items():
                idf = np.log10(self.total_doc_num) - np.log10(freq)
                self.idf[term_id] = idf
            
            print("building idf dictionary selesai....")
                
        except FileNotFoundError:
            print("index file not found!")

  def update_idf(self, token_id, new_len_postings_list):
      if token_id not in self.df:
        self.df[token_id] = new_len_postings_list
      else:
        self.df[token_id] += new_len_postings_list
      self.idf[token_id] = np.log10(self.total_doc_num) - np.log10(self.df[token_id]+1)
      

  def get_idf(self, term):
        smother =  np.log10(self.total_doc_num) - np.log10(1)
        return self.idf.get(term, smother)

 

  def compute_tf_idf(self, query: str):
        """
        membuat tf-idf scoring untuk setiap document dari query
        ref: https://web.stanford.edu/class/cs276/19handouts/lecture6-tfidf-6per.pdf
        
        compute tf idf untuk dynamic index & in memory index juga
        ref: "If there is a requirement that new documents be included quickly, one solution is to maintain two indexes: a large main index and a small auxiliary index that stores new documents. The auxiliary index is kept in memory. Searches are run across both indexes and results merged."
        from: https://nlp.stanford.edu/IR-book/html/htmledition/dynamic-indexing-1.html

        document yang didelete difilter/dihapus dari query result ( iterate docID di setiap posting)
        ref: " Deletions are stored in an invalidation bit vector. We can then filter out deleted documents before returning the search result. Documents are updated by deleting and reinserting them.". https://nlp.stanford.edu/IR-book/html/htmledition/dynamic-indexing-1.html

        """
        with InvertedIndex(self.index_name, directory=self.output_dir
                    ) as mapper:
            
            query = stemmer.stem(query)
            terms = nltk.word_tokenize(query)
            w_t_d = {}

            sortedDocumentScores = {}
            tf_per_term = {}
            for term in terms:
                term_id = self.term_id_map.str_to_id.get(term.lower())
                tf_per_term[term_id] = dict(dict())

            for term in terms:
                term_id = self.term_id_map.str_to_id.get(term.lower())
                postings_list = mapper[term_id]
                

                # menghitung tf untuk main inverted index
                for docID in postings_list:
                    if self.invalidation_bit_vector[docID] == True:
                        # document with docID deleted  
                        continue
                    if docID not in tf_per_term[term_id]:
                        tf_per_term[term_id][docID] =  1 / self.docWordCount[docID]
                    else:
                        tf_per_term[term_id][docID] += 1 / self.docWordCount[docID]


                in_memory_posting_list = []
                if term_id in self.in_memory_indices:
                    in_memory_posting_list = self.in_memory_indices[term_id]

                # menghitung tf untuk in-memory inverted index
                for docID in in_memory_posting_list:
                    if  self.invalidation_bit_vector[docID] == True:
                        # document with docID deleted 
                        continue
                    # document di in memory inverted index sama document di main inverted index beda
                    if docID not in tf_per_term[term_id]:
                        tf_per_term[term_id][docID] = 1 / self.docWordCount[docID] 
                    else:
                        tf_per_term[term_id][docID] += 1 / self.docWordCount[docID]

            # menghitung tf untuk setiap on-disk inverted index di self.indices
            for inverted_index in self.indexes:
                with InvertedIndex("DynamicSIPMI_BSBI_Lintang_"+str(inverted_index), directory=self.output_dir
                    ) as curr_index:
                        for term in terms:
                            term_id = self.term_id_map.str_to_id.get(term.lower())
                            postings_list = []
                            if term_id in curr_index.postings_dict:
                                postings_list = curr_index[term_id]

                            for docID in postings_list:
                                if self.invalidation_bit_vector[docID] == True:
                                    # document with docID deleted 
                                    continue
                                if docID not in tf_per_term[term_id]:
                                    tf_per_term[term_id][docID] =  1 / self.docWordCount[docID]
                                else:
                                    tf_per_term[term_id][docID] += 1 / self.docWordCount[docID]

             
            for term_id, docIDs in tf_per_term.items():
                for docID in docIDs.keys():
                    w_t_d[docID, term_id] =  (np.log10(tf_per_term[term_id][docID] +1)) *  self.idf[term_id]
        
            documentScores = {}
            
            for (docID, term_id), tfidf  in w_t_d.items():
                if docID not in documentScores:
                    documentScores[docID] = tfidf
                else:
                    documentScores[docID] += tfidf
        
            sortedDocumentScores = dict(sorted(documentScores.items(), key=lambda item: item[1], reverse=True))

        sortedRes = {}
        for docID, v in sortedDocumentScores.items():
            if v != 0 and len(sortedRes) <= 10:
                sortedRes[docID] = self.doc_id_map[docID]
        res = [(key, value) for key, value in sortedRes.items()]
        return res
  
  
  
