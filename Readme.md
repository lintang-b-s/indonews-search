# Indonews-fts
full text search database (on-disk) for indonesian news (or can be anything) from scratch. all documents in the corpus are inverted indexed (with on-disk postings list). Document indexing is done using dynamic indexing, so that new documents can be indexed into the database. currently the query only supports tf-idf scoring. By default the application indexing the following dataset: https://drive.google.com/file/d/1-AbtUsBbMQJ6qe_cDhjy4_S7D18Cw7ZX/view?usp=sharing


# Quick Start
### Run the app
```
- python -m venv venv
- source ./venv/bin/activate
- pip install -r requirements.txt
- flask --app app run
```
### Query
```
curl --location --request GET 'http://localhost:5000/' \
--header 'Content-Type: application/json' \
--data '{
    "query": "gibran dan jokowi lagi di gorong-gorong"
}'
```

### Indexing document
- indexing news document by "content"
```
    curl --location 'http://localhost:5000/index' \
--header 'Content-Type: application/json' \
--data '{
    "title": "Pemilik akun Kaskus Fufu Fafa dan Kaesang berperilaku kembar Siam Mesti Di Bypass Bahayakan NKRI",
    "content": " Tuduhan publik bahwa pemilik akun Kaskus FufuFafa yang isinya berisikan narasi menjijikan yang menghujat Prabowo Subianto (Presiden RI terpilih 2024) adalah Gibran RR. Sang terlapor gratifikasi dan atau money laundry (kriminal) juga amoral (putus urat malu) karena nafsu syahwatnya ingin turut serta dalam kompetisi pilpres 2024, walau usia tak mencukupi, namun memaksakan kehendak dengan pola disobidience/ pembangkangan terhadap sistim konstitusi, sehingga berimplikasi lahirkan historis hukum vonis MKMK. memberhentikan Anwar Usman sebagai Ketua Mahkamah Konstitusi/ MK karena terbukti melakukan nepotisme demi sang keponakan Gibran RR Kemudian berlanjut, deskripsi sosok Gibran di ruang debat cawapres 2024, ternyata dikecam banyak publik karena tunjukan perangai buruk (bad attitude). Sehingga gambaran akumulasi tipikal yang dipresentasikan Gibran didapati banyak temuan rekam jejak perilaku abnormal (bad atiltitude), karena amat tidak role model, sehingga pola kepemimpinan Gibran berpotensi timbulkan dampak buruk terhadap mentalitas anak bangsa, maka ungkapan melalui sarkastik akan pas, bahwa Gibran pemimpin berkarakter “BANGSAT BANGSA” Sedangkan perilaku adiknya Kaesang, kapasitasnya nyaris mirip Kakaknya (Gibran), yang memaksakan egonya melalui modus nepotisme dengan cara-cara pola disobidience, dengan tujuan ingin ikut Pilkada walau umurnya tidak berkesesuaian dengan sistim hukum. Persamaan lainnya, Kaesang pun terindikasi terlibat dibeberapa laporan kasus bersama Gibran sebagai penerima gratifikasi dan atau money laundry, serta juga terlapor sebagai penerima gratifikasi tiket gratis pesawat pribadi/ carteran ke Amerika bersama istrinya, berikut service kemudahan tanpa diperiksa melalui X-Ray di konter bea cukai bandara, namun Kaesang berkelit, gunakan argumentasi illogic, bahwa; “istrinya sedang hamil sehingga pelesiran harus naik pesawat pribadi / carteran berjenis Gulfstream G650ER bersama istrinya Erina Gudonoy lalu Kaesang pun terpaksa menerima gratifikasi atau gratis tiket Perbedaan antara kakak-adik ini, praktik nepotisme Gibran melalui MK. Sedangkan Kaesang melalui MA./ Mahkamah Agung. Maka kedua kakak beradik Gibran dan Kaesang keburukan perilakunya bak kembar siam. Sehingga keselamatan anak bangsa dan negara mesti prioritas, sebab efek titisan “moral hazard” asal genetika Jokowi, sukses mengalir turun kepada kedua putranya, sehingga Gibran-Kaesang harus di bypass melalui due process (rule of law) demi fungsi penegakan hukum (law enforcement), yakni manfaat hukum/ utilitas, sebagai efek jera dan tidak berlanjut mengkontaminasi anak bangsa), kepastian hukum/ legalitas, serta rasa keadilan/ justice. "
}'
```

- query new indexed doc
```
curl --location --request GET 'http://localhost:5000/' \
--header 'Content-Type: application/json' \
--data '{
    "query": "fufufafa bangsat"
}'
```


- indexing news doc by its "content"
```
curl --location 'http://localhost:5000/index' \
--header 'Content-Type: application/json' \
--data '{
    "title": "10 Hal Seru yang Dilakukan IShowSpeed di Yogyakarta, Jadi Mas-mas Jawa hingga Cicipi Beras Kencur",
    "content": "Ada banyak hal seru yang dilakukan IShowSpeed saat berkunjung ke Yogyakarta, Jawa Tengah pada Sabtu, 21 September 2024. Perjalanannya ini terekam dalam kanal YouTube pribadinya, IShowSpeed. IShowSpeed menikmati berbagai pengalaman khas lokal yang penuh keakraban dengan budaya Jawa dan kearifan Yogyakarta . Dengan gayanya yang khas, Youtuber asal Amerika ini pun berhasil mencuri perhatian netizen Indonesia. Bahkan, videonya saat berada di Yogyakarta sudah ditonton lebih dari 7,5 juta kali. Pria yang dipanggil El Kecepatan ini sebelumnya lebih dahulu mengunjungi Malaysia, kemudian menyambangi Jakarta, serta Bali dengan ditemani Reza Arap. Berikut adalah sederet keseruan IShowSpeed saat berkunjung ke Yogyakarta dikutip dari akun YouTube pribadinya, Minggu (22/9/2024). Saat sampai di Teras Malioboro, Speed langsung disambut oleh tiga orang yang berdandan ala tukang jamu gendong. Mereka kemudian memakaikan sang Youtuber baju batik dan blangkon, hingga menawarkannya untuk mencoba salah satu oleh-oleh khas Yogyakarta, yakni bakpia. Setelah dibuat kebingungan saat memakai batik dan menjajal bakpia, Speed lalu berjalan menghampiri sejumlah pemain gendang yang tampak kompak mengenakan batik. Ia mencoba ikut bermain gendang bersama mereka. Menariknya, di tengah-tengah aksinya itu, ia berkata, Kita harus memulai membuat band Indonesia"
}'
```

- querying new indexed news doc
```
curl --location --request GET 'http://localhost:5000/' \
--header 'Content-Type: application/json' \
--data '{
    "query": "Ishowspeed di Yogyakarta memakai batik"
}'
```

# Ref
```
- https://web.stanford.edu/class/cs276/19handouts/lecture2-intro-boolean-6per.pdf
- https://nlp.stanford.edu/IR-book/pdf/04const.pdf
- https://nlp.stanford.edu/IR-book/pdf/06vect.pdf
- https://web.stanford.edu/class/cs276/19handouts/lecture6-tfidf-6per.pdf
```
