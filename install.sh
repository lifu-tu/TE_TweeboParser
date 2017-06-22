ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARSER_DIR="${ROOT_DIR}/TBParser"

rm pretrained_models.tar.gz
curl "http://ttic.uchicago.edu/~lifu/TE_TweeboParser/pretrained_models.tar.gz" -o "pretrained_models.tar.gz"
tar xvf pretrained_models.tar.gz

curl "http://ttic.uchicago.edu/~lifu/TE_TweeboParser/wordvects.tw100w5-m40-it2" -o "wordvects.tw100w5-m40-it2"
mv wordvects.tw100w5-m40-it2 TE_Parser/embeddings/

cd ${PARSER_DIR}
chmod +x install-sh
./install_deps.sh
./configure && make && make install
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:`pwd;`/deps/local/lib:"
