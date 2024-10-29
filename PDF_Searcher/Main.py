import PyPDF2, pymupdf, urllib.request, nltk, textract
from io import BytesIO
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
"""
# Accessing our pdf file, making object of PYPDF2 class, bytesIO to convert pdf file to raw bytes,
# understood by PYPDF2
pdf_file = urllib.request.urlopen('https://www.i4n.in/wp-content/uploads/2023/05/Recipe-Book.pdf')
pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))

# An example to get page of the pdf, then extract text out of it
pageObj = pdf_reader.pages[16]
page2 = pageObj.extract_text()

# Tokenising all words in page to remove punctuations and stop words
punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
                '@', '[', ']', '^', '_', '`', '{', '|', '}', '~']
tokens = word_tokenize(page2)
stop_words = stopwords.words('english')
keywords = [word for word in tokens if word not in stop_words and word not in punctuations]

print(keywords)"""


#Implementing a more advanced library, Pymupdf


doc = pymupdf.open("https://www.i4n.in/wp-content/uploads/2023/05/Recipe-Book.pdf") # open a document
out = open("output.txt", "wb") # create a text output
for page in doc: # iterate the document pages
    text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
    out.write(text) # write text of page
    out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
out.close()
