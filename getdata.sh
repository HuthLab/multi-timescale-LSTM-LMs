echo "=== Acquiring datasets ==="

mkdir -p save

mkdir -p data
cd data

echo "- Downloading Penn Treebank (PTB)"
#if on Mac OS use following
echo "Using curl for MacOS. If you are on Linux, please use wget command commented below." #OR if on Linux use following command
#wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

curl --silent http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz --output simple-examples.tgz

tar -xf simple-examples.tgz

mkdir -p penn
cd penn
mv ../simple-examples/data/ptb.train.txt train.txt
mv ../simple-examples/data/ptb.test.txt test.txt
mv ../simple-examples/data/ptb.valid.txt valid.txt


cd ..

rm -rf simple-examples , simple-examples.tgz
echo " For downloading WikiText-2 (WT2): Change 'if' condition below to 'true'. "
if false; then
#if on Linux use following command
#wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
# OR if on Mac OS use following

curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip
cd wikitext-2
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
fi

echo "---"
echo "Happy language modeling :)"
