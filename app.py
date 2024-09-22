import os
import  urllib
if os.path.exists("./News.csv") == False:
        print("mendownload file News.csv....")
        url = "https://drive.google.com/uc?export=download&id=1-AbtUsBbMQJ6qe_cDhjy4_S7D18Cw7ZX"
        path = "News.csv"
        urllib.request.urlretrieve(url, path)
        print("selesai mendownload file News.csv")
from fts.index_constructor import DynamicBSBIIndexer

from flask import Flask, request, jsonify

app = Flask(__name__)

if os.path.exists("./output_dir") == False:
        os.mkdir("./output_dir")


with app.app_context():
    global BSBI_instance
    BSBI_instance = DynamicBSBIIndexer(file_path= "./News.csv", output_dir = 'output_dir',inverted_index_buffer_size=1e8)
    for (_, _, filenames) in os.walk("./output_dir"):
         if len(filenames) == 0:
             BSBI_instance.index()
    BSBI_instance.build_idf()





@app.route("/", methods=['GET'])
def query():
    if request.method == 'GET':
        req_data = request.get_json()
        query = req_data['query']
        results = BSBI_instance.compute_tf_idf(query=query)
        return jsonify({
                    'res': results,
                    'status': '200',
                    'msg': 'Success'
                })

@app.route("/index", methods=['POST'])
def index_doc():
     if request.method == 'POST':
          req_data = request.get_json()
          title = req_data['title']
          content = req_data['content']
          BSBI_instance.lMergeAddToken(content, title)
          return jsonify({
                    'status': '200',
                    'msg': 'Success indexing this news'
                })


if __name__ == '__main__':  
    app.run(use_reloader=False)

