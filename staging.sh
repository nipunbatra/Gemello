wget https://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/anaconda
export PATH="$HOME/anaconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda install -c conda
conda install --yes pip
mkdir -p git
cd git
wget --no-check-certificate https://github.com/nipunbatra/Gemello/tarball/master
tar -zvxf master
mv nipunbatra-* Gemello
cd Gemello/cluster
python cluster_create_inequalities_small.py
