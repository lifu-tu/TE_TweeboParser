# Copyright (c) 2013-2014 Lingpeng Kong
# All Rights Reserved.
#
# This file is part of TweeboParser 1.0.
#
# TweeboParser 1.0 is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TweeboParser 1.0 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with TweeboParser 1.0.  If not, see <http://www.gnu.org/licenses/>.

# This script runs the whole pipeline of TweeboParser. It should install the 
# TurboParser inside the TweeboParser for you.

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
