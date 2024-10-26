import PyPDF2, urllib.request, nltk, textract
from io import BytesIO
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')

# Accessing our pdf file, making object of PYPDF2 class, bytesIO to convert pdf file to raw bytes,
# understood by PYPDF2
pdf_file = urllib.request.urlopen('https://www.i4n.in/wp-content/uploads/2023/05/Recipe-Book.pdf')
pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))

# An example to get page of the pdf, then extract text out of it
pageObj = pdf_reader.pages[2]
page2 = pageObj.extract_text()

# Tokenising all words in page to remove punctuations and stop words
punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
                '@', '[', ']', '^', '_', '`', '{', '|', '}', '~']
tokens = word_tokenize(page2)
stop_words = stopwords.words('english')
keywords = [word for word in tokens if not word in stop_words and not word in punctuations]

print(tokens)
print(keywords)

