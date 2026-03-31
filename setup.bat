@echo off
echo Installing required Python packages...
:: Python 3.14 does not yet have pre-built wheels for torch/gensim.
py -m pip install pandas numpy scikit-learn nltk matplotlib seaborn

echo Packages installed successfully!
echo Downloading NLTK resources...
py -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
echo Setup complete. You can now run the main script.
pause
